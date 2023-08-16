#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-14 下午 4:30
# @Author     : zhangyb
# @File       : test.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
import time

import torch

from loguru import logger as log


def callback(step: int, timestep: int, latents: torch.FloatTensor):
    log.info(f'step: {step}')
    log.info(f'timestep: {timestep}')
    log.info(f'latens: {latents}')


# pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
#     "./models/sd_xl_refiner_1.0.safetensors", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# print(pipe)
# pipe = pipe.to("cuda")
# pipeline_setter(pipe)
#
# url = "./files/input_files/demo6351.png"
# init_image = load_image(url).convert("RGB")
# from optimum.intel import OVStableDiffusionXLPipeline

# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id, cache_dir='../models', local_files_only=True)
# prompt = "sailing ship in storm by Rembrandt"
# image = pipeline(prompt).images[0]

# prompt = "a photo of an astronaut riding a horse on mars"

# image = pipe(prompt, image=init_image, strength=0.5, num_inference_steps=20, callback=callback).images[0]
# image.save(f"files/output_files/demo_{int(time.time())}.png")
