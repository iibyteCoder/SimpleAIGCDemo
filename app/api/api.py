#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-14 下午 6:51
# @Author     : zhangyb
# @File       : api.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
from fastapi import APIRouter
from fastapi.params import Query
from loguru import logger as log

from app.core.others import encode_pil_to_base64
from app.core.progress_tracing import TaskTrack
from app.core.settings import get_setting_field
from app.core.text_to_image import TextToImageProcessor
from app.models.models import Txt2ImgRequestItem, Img2ImgRequestItem, ResponseItem

router = APIRouter(
    prefix='/AIGC/v1',
    responses={404: {"description": "Not found"}},
)


@router.post("/txt2img", tags=['文生图'], description='文生图', response_model=ResponseItem)
# @logger
def txt2img(request_item: Txt2ImgRequestItem):
    log.info(request_item)
    width = request_item.width
    height = request_item.height
    prompt = request_item.prompt
    negative_prompt = request_item.negative_prompt
    steps = request_item.steps
    scale = request_item.scale
    size = request_item.size
    task_index = request_item.task_index
    nprompt = get_setting_field('diffusers.negative_prompt')
    tracker = TaskTrack(task_index=task_index, total_steps=steps)
    process = TextToImageProcessor(prompt=prompt, negative_prompt=negative_prompt or nprompt, width=width,
                                   height=height, steps=steps, scale=scale, size=size, tracker=tracker)

    future = process.submit_task()
    result = future.result()
    images_list = result.images
    b64images = list(map(encode_pil_to_base64, images_list))
    log.info(b64images)
    return ResponseItem(images=b64images, )


@router.post("/img2img", tags=['图生图'], description='图生图', response_model=ResponseItem)
# @logger
def img2img(request_item: Img2ImgRequestItem):
    log.info(request_item)
    return {"message": "Hello World"}


@router.get('/interrupt')
def interrupt(index: str = Query(...)):
    tracker = TaskTrack.state_map.get(index)
    response = ResponseItem(task_index=index)
    if tracker:
        try:
            tracker.task.cancel()
        except Exception as e:
            log.error(f'打断发生错误，task_index：{tracker.task_index}', e)
    response.info = {"success": True}
    return response


@router.get('/progress')
def progress(index: str = Query(...)):
    tracker = TaskTrack.state_map.get(index)
    response = ResponseItem(task_index=index)
    if tracker:
        response.info = {"success": True, "total_steps": tracker.total_steps, "current_step": tracker.current_step}
        response.images = tracker.images
        return response
    response.info = {"message": "任务尚未开始"}
    return response
