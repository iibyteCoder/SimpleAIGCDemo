#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-14 下午 5:19
# @Author     : zhangyb
# @File       : progress_tracing.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
from concurrent.futures import Future

import torch
from loguru import logger as log


class TaskTrack:
    state_map: dict = {}

    """
    每个任务的状态记录
    """

    def __init__(self, task_index, total_steps, pipeline=None, current_step=None, images=None,
                 task=None) -> None:
        super().__init__()
        self.task_index = task_index
        self.total_steps = total_steps
        self.current_step = current_step
        self.images: list = images
        self.pipeline = pipeline
        self.task: Future = task

    def put(self):
        self.state_map[self.task_index] = self

    def image_callback(self, i: int, t: int, latents: torch.FloatTensor):
        """图像处理过程中当前状态、图片设置"""
        log.info(f'step: {i} | timestep: {t}')
        self.current_step = i
        latents = 1 / 0.18215 * latents
        image = self.pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        self.images.extend(self.pipeline.numpy_to_pil(image))

    def thread_callback(self):
        """线程完成之后弹出"""
        # self.state_map.pop(self.task_index)
        ...
