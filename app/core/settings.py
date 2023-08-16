#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-15 上午 11:54
# @Author     : zhangyb
# @File       : settings.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
import os.path

import toml

# 各种路径
app_dir = os.path.dirname(os.path.dirname(__file__))
config_file_path = os.path.join(app_dir, 'config.toml')
root_dir = os.path.dirname(app_dir)
model_path = os.path.join(root_dir, 'models')

# 配置文件
config = toml.load(config_file_path)


def get_setting_field(field_name: str):
    str_list = field_name.split('.')
    layer = config
    for field in str_list:
        layer = layer.get(field)
    return layer


if __name__ == '__main__':
    print(root_dir)
    # print(config)
    # print(config.get('diffusers').get('negative_prompt'))
    print(get_setting_field('app.thread_count'))
