#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-16 下午 4:30
# @Author     : zhangyb
# @File       : test4.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
import multiprocessing
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import StableDiffusionXLPipeline

from app.core.settings import model_path, files_path
from loguru import logger as log

sd = StableDiffusionXLPipeline.from_single_file(os.path.join(model_path, 'sd_xl_base_1.0.safetensors'),
                                                use_safetensors=True, cache_dir=model_path, local_files_only=True,
                                                torch_dtype=torch.float16)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"


def run_inference(rank, world_size):
    log.info(f'>>>>>>>>>>>> | {random.random()}')
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    sd.to(rank)
    prompt = ''
    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    image = sd(prompt).images[0]
    file_path = os.path.join(files_path, f"output_files/{'_'.join(prompt)}.png")
    image.save(file_path)


def main():
    world_size = 1
    b = mp.spawn(run_inference, args=(1,), nprocs=world_size, join=False)
    # a = mp.spawn(run_inference, args=(1,), nprocs=world_size, join=True)
    log.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    b.join()
    # a.join()


if __name__ == "__main__":
    main()
