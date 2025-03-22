from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torch
import cv2
import random
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor

augment=Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    img = img * 2.0 - 1.0
    return img

class PairedImageDataset(Dataset):
    def __init__(self, data_dir,patchsize=None,aug=True,training=True):
        self.data_dir = data_dir
        self.target_dir=self.data_dir+'/target'
        self.input_dir=self.data_dir+'/input'
        # 获取 input 和 target 目录下的文件名
        self.low_images = sorted([f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))])
        self.high_images = sorted([f for f in os.listdir(self.target_dir) if os.path.isfile(os.path.join(self.target_dir, f))])
        self.dir_size = len(self.low_images) 
        self.aug=aug
        self.crop_size=patchsize
        self.train=training
        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        # 返回数据集的大小.target_dir
        return self.dir_size

    def __getitem__(self, idx):
        # 根据索引获取文件名
        low_image_path = os.path.join(self.input_dir, self.low_images[idx])
        high_image_path = os.path.join(self.target_dir, self.high_images[idx])
        ps=self.crop_size
        # 打开 input 和 target 图像
        noisy  = torch.from_numpy(np.float32(load_img(low_image_path)))
        clean  = torch.from_numpy(np.float32(load_img(high_image_path)))

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        if self.train and self.crop_size:
            ps = self.crop_size
            H = clean.shape[1]
            W = clean.shape[2]

            if H-ps==0:
                r=0
                c=0
            else:
                r = np.random.randint(0, H - ps)
                c = np.random.randint(0, W - ps)

            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps]
            if self.aug==True:
                apply_trans = transforms_aug[random.getrandbits(3)]
                clean = getattr(augment, apply_trans)(clean)
                noisy = getattr(augment, apply_trans)(noisy)   
        return clean, noisy







def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = PairedImageDataset(
        data_dir,
        image_size
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
