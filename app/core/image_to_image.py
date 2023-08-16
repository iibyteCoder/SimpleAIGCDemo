#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-08-14 下午 4:13
# @Author  : zhangyb
# @File    : image_to_image.py
# @Software: PyCharm
import os.path
import time

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from loguru import logger as log

from app.core.pipeline_generator import pipeline_setter
from app.core.settings import model_path
from others import get_generator, timer


@timer
def get_img2img_pipeline(
        model=os.path.join(model_path, 'sd_xl_refiner_1.0.safetensors'),
        cache_dir=model_path,
        use_safetensors=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        variant="fp16"
) -> StableDiffusionXLImg2ImgPipeline:
    """
    获取StableDiffusionXLPipeline
    用于文生图
    """
    pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model,
        cache_dir=cache_dir,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        variant=variant
    )
    pipeline_setter(pipeline)
    return pipeline


@timer
def image_to_image(
        image,
        pipeline,
        prompt,
        negative_prompt,
        # width,
        # height,
        scale,
        steps,
        generator,
        strength=0.7,
        num_images_per_prompt=1,
):
    return pipeline(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        # width=width,
        # height=height,
        guidance_scale=scale,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )


if __name__ == '__main__':
    pipeline = get_img2img_pipeline()
    print(pipeline)
    width = 512 * 2
    height = 512 * 2
    generator = get_generator()
    image_path = '../../files/input_files/demo49409.png'
    image = load_image(image_path).convert('RGB')
    ####################################################################
    prompt = 'puppy'
    log.info(f'>>>>>| {prompt} |<<<<<')
    pipe_out = image_to_image(
        image=image,
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt="NSFW",
        # width=width,
        # height=height,
        steps=20,
        scale=7,
        generator=generator
    )
    print(pipe_out)
    print(pipe_out.__init__)
    image = pipe_out.images[0]
    image.save(f"files/output_files/demo_{int(time.time())}.png")
