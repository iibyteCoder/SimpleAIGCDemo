#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-08-14 下午 3:41
# @Author  : zhangyb
# @File    : others.py
# @Software: PyCharm
import base64
import io
import platform
import time
from io import BytesIO

import torch
from PIL import PngImagePlugin
from loguru import logger as log


def linux_or_not() -> bool:
    system = platform.system()
    return 'linux' in system.lower()


def to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64_str = base64.b64encode(buffer.getvalue())
    return img_base64_str


def to_base64_2(file_path: str):
    with open(file_path, 'rb') as ff:
        base64_binary_str = base64.b64encode(ff.read())
        base64_str = base64_binary_str.decode()
        return base64_str


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        metadata = PngImagePlugin.PngInfo()
        image.save(output_bytes, format="PNG", pnginfo=metadata)
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data)


def to_file(base64_str, file):
    ori_image_data = base64.b64decode(base64_str)
    with open(file, 'wb') as out:
        out.write(ori_image_data)


def get_generator(seed=int(time.time())):
    return torch.Generator(device="cuda").manual_seed(seed)


def timer(func):
    """
    计时
    """

    def wrapper(*args, **kwargs):
        t1 = time.time()
        log.info(f'{func.__name__:>20} | start >>>')
        res = func(*args, **kwargs)
        log.info(f'{func.__name__:>20} | end <<< | 耗时：{time.time() - t1:.3}s')
        return res

    return wrapper


def logger(func):
    """
    打印参数
    """

    def wrapper(*args, **kwargs):
        t1 = time.time()
        log.info(f'>>> {func.__name__} | {locals()} >>>')
        res = func(*args, **kwargs)
        return res

    return wrapper


if __name__ == '__main__':
    # file = '../../files/input_files/demo6351.png'
    # base64_str = to_base64_2(file)
    # print(base64_str)
    vae = pipe.vae
    images = []


    def latents_callback(i, t, latents):
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        images.extend(pipe.numpy_to_pil(image))


    prompt = "Portrait painting of Jeremy Howard looking happy."
    torch.manual_seed(9000)
    final_image = pipe(prompt, callback=latents_callback, callback_steps=12).images[0]
    images.append(final_image)
    image_grid(images, rows=1, cols=len(images))
