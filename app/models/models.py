#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-14 下午 6:51
# @Author     : zhangyb
# @File       : models.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
import random
import time
from typing import Dict, List

from fastapi import UploadFile
from fastapi.params import File
from pydantic import BaseModel, Field
from starlette.datastructures import FormData


class RequestItem(BaseModel):
    prompt: str = Field(default=None, title="正向提示词", description="生成照片所需要包含的描述")
    negative_prompt: str | None = Field(default=None, title="负向提示词", description="生成图片需要排除因素的描述")
    steps: int = Field(default=20, title="步数", description="设置生成图片的步数，步数越大图片越精细", ge=5, le=50)
    scale: float = Field(default=0.7, title='符合度', description='生成照片符合提示词的程度', ge=0, le=1)
    size: int = Field(default=1, title='数量', description='一次生成的图片数量')
    task_index: str = Field(default=time.time(), title='任务编号', description='用于查找任务当前状态')


class Txt2ImgRequestItem(RequestItem):
    prompt: str = Field(default=None, title="正向提示词", description="生成照片所需要包含的描述")
    width: int = Field(default=512, title="宽度", description="生成照片的宽度")
    height: int = Field(default=512, title="高度", description="生成照片的高度")


class Img2ImgRequestItem(RequestItem):
    image: str = Field(default=None, title='图片', description='图片文件base64编码', )
    strength: float = Field(default=0.5, title="重绘因子", ge=0.0, le=1.0,
                            description="绘制图片过程中添加噪声的程度，在图生图中影响图片重绘程度，"
                                        "具体表现为影响与steps参数一同决定实际执行的步数")


class ResponseItem(BaseModel):
    task_index: str = Field(default=time.time().__str__().replace('.', ''),
                            title='任务编号', description='用于返回本次请求生成的任务编号')
    images: List[bytes] = Field(default=None, title='图片（base64格式）', description='生成的图片，以base64编码')
    parameters: dict = Field(default=None, title='参数')
    info: Dict = Field(default=None, title='其他信息', description='其他信息')


if __name__ == '__main__':
    print(time.time().__str__().replace('.', ''))
