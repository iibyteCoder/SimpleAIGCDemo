#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-14 下午 5:19
# @Author     : zhangyb
# @File       : progress_tracing.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
import os.path
from concurrent.futures import Future

import torch
from loguru import logger as log

from app.core.others import param_log
from app.core.settings import files_path


class TaskTrack:
    state_map: dict = {}

    """
    每个任务的状态记录
    """

    def __init__(self, task_index, total_steps, pipeline=None, current_step=None, task=None) -> None:
        super().__init__()
        self.task_index = task_index
        self.total_steps = total_steps
        self.current_step = current_step
        self.time_left: float | None = None
        self.images: list = []
        self.pipeline = pipeline
        self.task: Future = task
        self.interrupted = False

    def put(self):
        self.state_map[self.task_index] = self

    def image_callback_bak(self, i: int, t: int, latents: torch.FloatTensor):
        """图像处理过程中当前状态、图片设置"""
        if self.interrupted:
            raise InterruptedError(f'生成任务中断，任务编号：{self.task_index}')
        log.info(f'step: {i} | timestep: {t}')
        self.current_step = i
        self.time_left = t
        latents = 1 / 0.18215 * latents
        image = self.pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        self.images.extend(self.pipeline.numpy_to_pil(image))

    def image_callback(self, step, time, latents):
        if self.interrupted:
            raise InterruptedError(f'生成任务中断，任务编号：{self.task_index}')
        log.info(f'step: {step} | timestep: {time}')
        self.current_step = step
        self.time_left = time
        # convert latents to image
        latents = 1 / 0.18215 * latents
        image = self.pipeline.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # convert to PIL Images
        image = self.pipeline.numpy_to_pil(image)
        self.images.extend(image)
        # do something with the Images
        for i, img in enumerate(image):
            output_file = os.path.join(files_path, 'output_files', f"iter_{step}_img{i}.png")
            img.save(output_file)

    @param_log
    def thread_callback(self, *args, **kwargs):
        """线程完成之后弹出"""
        # self.state_map.pop(self.task_index)
        ...
