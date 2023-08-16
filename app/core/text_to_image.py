#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-08-14 上午 10:22
# @Author  : zhangyb
# @File    : text_to_image.py
# @Software: PyCharm
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from loguru import logger as log

from app.core.others import get_generator, timer
from app.core.pipeline_generator import PipelineContainer
from app.core.progress_tracing import TaskTrack
from app.core.settings import get_setting_field

# use the TensorFloat32 (TF32) mode for faster but slightly less accurate computations
torch.backends.cuda.matmul.allow_tf32 = True
thread_count = get_setting_field('app.thread_count')
executor = ThreadPoolExecutor(max_workers=thread_count)


class TextToImageProcessor:

    def __init__(self, prompt, negative_prompt, width, height, scale, steps, tracker: TaskTrack, size=1) -> None:
        super().__init__()
        self.pipeline_container = PipelineContainer()
        self.generator = get_generator()
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.scale = scale
        self.steps = steps
        self.num_images_per_prompt = size
        self.tracker = tracker

    @timer
    def text_to_image(self):
        pipeline = self.pipeline_container.get()
        self.tracker.pipeline = pipeline
        try:
            return pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                width=self.width,
                height=self.height,
                guidance_scale=self.scale,
                num_images_per_prompt=self.num_images_per_prompt,
                generator=self.generator,
                output_type="pil",
                callback=self.tracker.image_callback
            )
        except InterruptedError as i:
            log.info(i)
            raise InterruptedError(*i.args)
        except Exception as e:
            log.error(e)
        finally:
            log.info(f'------- 放回队列中：{pipeline.__class__}：{id(pipeline)} --------')
            self.pipeline_container.put(pipeline=pipeline)

    # def process(self, prompt, width, height):
    #     # pipeline = get_txt2img_pipeline()
    #     # generator = get_generator()
    #     pipe_out = self.text_to_image(
    #         prompt=prompt, negative_prompt="NSFW", width=width, height=height, steps=20, scale=7
    #     )
    #     image = pipe_out.images[0]
    #     # image.save(f"files/output_files/demo_{time.time()}.png")
    #     return image

    def submit_task(self) -> Future:
        task = executor.submit(self.text_to_image, )
        task.add_done_callback(self.tracker.thread_callback)
        self.tracker.task = task
        self.tracker.put()
        return task


if __name__ == '__main__':
    # base_pipeline = get_base_pipeline(local_files_only=False, cache_dir='./models')
    # image = imagine(base_pipeline, prompt='puppy', negative_prompt="NSFW", width=512, height=512, scale=7, steps=20,
    #                 seed=random.randint())
    # image.save("files/output_files/demo.png")
    # pipeline = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
    #                                                    local_files_only=False, cache_dir='./models')
    # pipeline = get_txt2img_pipeline()
    width = 512 * 2
    height = 512 * 2
    # generator = get_generator()

    ####################################################################
    # prompt = 'puppy'
    # log.info(f'>>>>>{prompt}<<<<<')
    # pipe_out = text_to_image(pipeline=pipeline, prompt=prompt, negative_prompt="NSFW", width=width, height=height,
    #                          steps=20, scale=7, generator=generator)
    # image = pipe_out.images[0]
    # image.save(f"files/output_files/demo_{random.randint(0, 99999)}.png")
    # log.info('------next------')
    # prompt = 'pussy'
    # log.info(f'>>>>>{prompt}<<<<<')
    # pipe_out = text_to_image(pipeline=pipeline, prompt=prompt, negative_prompt="NSFW", width=width, height=height,
    #                          steps=20, scale=7, generator=generator)
    # image = pipe_out.images[0]
    # image.save(f"files/output_files/demo_{random.randint(0, 99999)}.png")
    ####################################################################

    # a1 = TextToImageProcessor()
    # res1 = executor.submit(a1.process, prompt="panda", width=128, height=128)
    # # res2 = executor.submit(process, prompt='panda', width=1024, height=1024)
    # res1.result()
    # log.info(f'res1:{res1}')
    #
    # a2 = TextToImageProcessor()
    # res2 = executor.submit(a2.process, prompt="polar bear", width=128, height=128)
    # res2.result()
    # log.info(f'res2:{res2}')

    from pathos.pp import ParallelPool

    tracker = TaskTrack(task_index='666', total_steps=10)
    a3 = TextToImageProcessor(prompt='puppy', negative_prompt='nsfw', width=64, height=64, scale=0.7, steps=10,
                              tracker=tracker)
    # task = a3.submit_task()
    pool = ParallelPool(nodes=1)
    # pool = ProcessingPool(nodes=1)
    result = pool.map(a3.text_to_image, ('',))
    # print(result)
