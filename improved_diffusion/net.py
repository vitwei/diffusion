import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod
import ipdb
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding




class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            GroupNorm32(32,channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32,self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_channels=32,
        out_channels=3,
        dropout=0,
    ):
        super().__init__()
        self.in_channels=in_channels
        self.model_channels=model_channels
        self.out_channels=out_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )


        self.down_blocks=nn.ModuleList([])
        self.up_blocks=nn.ModuleList([])

        self.input_blocks = TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))

        self.down_blocks.append(TimestepEmbedSequential(ResBlock(model_channels,time_embed_dim,dropout,out_channels=model_channels),
                                                        ResBlock(model_channels,time_embed_dim,dropout,out_channels=model_channels),
                                                        Downsample(model_channels)))

        self.down_blocks.append(TimestepEmbedSequential(ResBlock(model_channels,time_embed_dim,dropout,out_channels=2*model_channels),
                                                        ResBlock(2*model_channels,time_embed_dim,dropout,out_channels=2*model_channels),
                                                        Downsample(2*model_channels)))
        
        self.down_blocks.append(TimestepEmbedSequential(ResBlock(2*model_channels,time_embed_dim,dropout,out_channels=4*model_channels),
                                                        ResBlock(4*model_channels,time_embed_dim,dropout,out_channels=4*model_channels),
                                                        Downsample(4*model_channels)))
        self.down_blocks.append(TimestepEmbedSequential(ResBlock(4*model_channels,time_embed_dim,dropout,out_channels=8*model_channels),
                                                        ResBlock(8*model_channels,time_embed_dim,dropout,out_channels=8*model_channels)))

        self.middle_block = TimestepEmbedSequential(
            ResBlock(8*model_channels,time_embed_dim,dropout,out_channels=8*model_channels),
            ResBlock(8*model_channels,time_embed_dim,dropout,out_channels=8*model_channels),
        )                                                   
                                                    #256+256->512->128
        self.up_blocks.append(TimestepEmbedSequential(ResBlock(16*model_channels,time_embed_dim,dropout,out_channels=16*model_channels),
                                                        ResBlock(16*model_channels,time_embed_dim,dropout,out_channels=4*model_channels)))
                                                    #128+128->256->64
        self.up_blocks.append(TimestepEmbedSequential(Upsample(8*model_channels),
                                                        ResBlock(8*model_channels,time_embed_dim,dropout,out_channels=8*model_channels),
                                                        ResBlock(8*model_channels,time_embed_dim,dropout,out_channels=2*model_channels)
                                                        ))
                                                    #64+64->128->32,128,128
        self.up_blocks.append(TimestepEmbedSequential(Upsample(4*model_channels),
                                                      ResBlock(4*model_channels,time_embed_dim,dropout,out_channels=4*model_channels),
                                                        ResBlock(4*model_channels,time_embed_dim,dropout,out_channels=model_channels)
                                                        ))
                                                    #32+32->64->32
        self.up_blocks.append(TimestepEmbedSequential(Upsample(2*model_channels),
                                                      ResBlock(2*model_channels,time_embed_dim,dropout,out_channels=2*model_channels),
                                                        ResBlock(2*model_channels,time_embed_dim,dropout,out_channels=model_channels)))

        self.out = nn.Sequential(
            GroupNorm32(32, model_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype



    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        E = x.type(self.inner_dtype)
        E = self.input_blocks(E, emb)
        hs.append(E)
        for module in self.down_blocks:
            E = module(E, emb)
            hs.append(E)
        E = self.middle_block(E, emb)
        for module in self.up_blocks:
            cat_in = torch.cat([E, hs.pop()], dim=1)
            E = module(cat_in, emb)
        E = E.type(x.dtype)
        return self.out(E)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result
    

def network_parameters(nets):
    num_params = sum(param.numel() for param in nets.parameters())
    return num_params

if __name__ == '__main__':
    net=UNetModel().cuda()
    input=torch.rand(1,3,256,256).cuda()
    timestep=torch.full((1,), 1, device='cuda', dtype=torch.long)
    pred_noise = net(input, timestep)
    print(network_parameters(net))