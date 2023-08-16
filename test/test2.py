#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2023-08-16 上午 11:04
# @Author     : zhangyb
# @File       : test2.py
# @ProjectName: SimpleAIGCDemo
# @Software   : PyCharm
import threading
from concurrent.futures import ThreadPoolExecutor
import time


def worker(task_id, stop_event):
    while not stop_event.is_set():
        print(f"Task {task_id} is running")
        time.sleep(1)
    print(f"Task {task_id} is stopped")


def main():
    executor = ThreadPoolExecutor(max_workers=2)
    stop_event = threading.Event()

    # Submit tasks
    tasks = [executor.submit(worker, i, stop_event) for i in range(3)]

    # Let tasks run for a while
    time.sleep(3)

    # Stop a specific task
    task_to_stop = tasks[1]
    stop_event.set()

    # Wait for tasks to complete
    executor.shutdown()


if __name__ == "__main__":
    main()
