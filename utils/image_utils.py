from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor

import numpy as np
import torch
import cv2
import os
import math


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


# =======================================
# get image pathes of files
# =======================================

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


# =======================================
# Conversion functions
# =======================================

def uint8_to_f32(img):
    return np.float32(img/255.)


def f32_to_uint8(img):
    return np.uint8((img.clip(0, 1)*255.).round())


def np3_to_tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)


def tensor4_to_np3(img):
    return img.permute(2, 3, 1, 0).squeeze().cpu().numpy()


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


# ---------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# ---------------------------------------------
def imread_np3_uint8(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    else:
        raise ValueError("Number of channels should be 1 or 3")
    return img


def imread_tensor4_f32(path, n_channels=3):
    img = imread_np3_uint8(path, n_channels)
    return np3_to_tensor4(uint8_to_f32(img))


def imsave_np3_uint8(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def imsave_tensor4_f32(img, path):
    img = f32_to_uint8(tensor4_to_np3(img))
    imsave_np3_uint8(img, path)


def imread(path, n_channels=3):
    return imread_tensor4_f32(path, n_channels)


def imsave(img, path):
    imsave_tensor4_f32(img, path)


# ---------------------------------------------
# Metrics for np3 image format
# ---------------------------------------------
# ----------
# PSNR
# ----------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ----------
# SSIM
# ----------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ---------------------------------------------
# Processing tools for Tensor4 image format
# ---------------------------------------------

def augment_img_tensor4(image: Tensor, mode=0):
    if mode == 0:
        return image
    elif mode == 1:
        return image.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return image.flip([2])
    elif mode == 3:
        return image.rot90(3, [2, 3])
    elif mode == 4:
        return image.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return image.rot90(1, [2, 3])
    elif mode == 6:
        return image.rot90(2, [2, 3])
    elif mode == 7:
        return image.rot90(3, [2, 3]).flip([2])


def convert_channels_number(image: Tensor, target_num_channels: int) -> Tensor:
    """
    Converts the number of channels of the image to the target number:
        - 1 channel -> n>1 channels: duplicates the channel
        - 3 channel -> 1 channel: RGB to gray conversion
        - n channels -> n channels: input image returned directly.
        - Other cases raise an error.
    :param image: input image Tensor with 4 dimensions (channel dimension at index 1).
    :param target_num_channels:
    :return: output image Tensor.
    """
    if image.shape[1] == target_num_channels:
        return image
    elif image.shape[1] == 1:
        return image.expand([-1, 3, -1, -1])
    elif image.shape[1] == 3 and target_num_channels == 1:
        return rgb2gray(image)
    else:
        raise ValueError(f'The number of input channels should be 1 or 3: found {image.shape[1]} channels.')


def rgb2gray(image: Tensor):
    return torch.sum(image * torch.tensor([[[[.2126]], [[.7152]], [[.0722]]]]), dim=1, keepdim=True)


def view_as_tensor4(t: Tensor) -> Tensor:
    shape = t.shape
    if len(shape) > 4:
        raise ValueError(f'The tensor has {len(shape)} dimensions. '
                         f'At most 4 are allowed to reshape to tensor4 image data.')
    return t.view(list(shape)+[1]*(4-len(shape)))


def modcrop(image: Tensor, scale: int):
    H, W = image.shape[2:4]
    H_r, W_r = H % scale, W % scale
    return image[:, :, :H - H_r, :W - W_r]
