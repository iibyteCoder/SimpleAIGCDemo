#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-15 下午 12:32
# @Author     : zhangyb
# @File       : pipeline_generator.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
import os
from queue import Queue

import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from app.core.others import linux_or_not, timer
from app.core.settings import get_setting_field, model_path


@timer
def pipeline_setter(pipeline):
    pipeline.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    pipeline.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    pipeline.to('cuda')
    # Sliced VAE decode for larger batches
    pipeline.enable_vae_slicing()
    # Tiled VAE decode and encode for large images
    pipeline.enable_vae_tiling()
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    # Windows not support yet
    if linux_or_not():
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)


@timer
def get_txt2img_pipeline(
        model=os.path.join(model_path, 'sd_xl_base_1.0.safetensors'),
        cache_dir=model_path,
        use_safetensors=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        variant="fp16"
) -> StableDiffusionXLPipeline:
    """
        获取StableDiffusionXLPipeline
        用于文生图
        """
    pipeline = StableDiffusionXLPipeline.from_single_file(
        model,
        cache_dir=cache_dir,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        variant=variant
    )
    return pipeline


class PipelineContainer:
    thread_count = get_setting_field('app.thread_count')
    queue = Queue(maxsize=thread_count)

    for i in range(thread_count):
        p = get_txt2img_pipeline()
        pipeline_setter(p)
        queue.put(p)

    def put(self, pipeline):
        # pipeline = get_txt2img_pipeline()
        self.queue.put(pipeline)

    def get(self):
        return self.queue.get(block=True)


if __name__ == '__main__':
    ...
