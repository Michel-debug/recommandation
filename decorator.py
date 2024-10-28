import time
import logging
from functools import wraps
import psutil
import os

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args,**kwargs):
        start_time = time.perf_counter()
        result = func(*args,**kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.info(f'function {func.__name__} run_time: {total_time:.4f} s')
        return result
    return timeit_wrapper

## This function is bad for multiprocessing, only for single process
def measure_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 定义获取整个进程树内存使用的函数
        def get_process_tree_memory_usage():
            process = psutil.Process(os.getpid())
            mem_usage = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    mem_usage += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            return mem_usage / 1024 / 1024  # 转换为 MB

        # 获取执行前的总内存使用
        mem_before = get_process_tree_memory_usage()
        # 执行函数
        result = func(*args, **kwargs)
        # 获取执行后的总内存使用
        mem_after = get_process_tree_memory_usage()
        # 计算内存差异
        mem_diff = mem_after - mem_before
        logging.info(f"Function '{func.__name__}' total memory usage:")
        logging.info(f"Before run: {mem_before:.2f} MB")
        logging.info(f"After run: {mem_after:.2f} MB")
        logging.info(f"Memory increased by: {mem_diff:.2f} MB")
        return result
    return wrapper

def printLog(func):
    @wraps(func)
    def log_wrapper(*args, **kwargs):
        func_name = func.__name__
        logging.info(f"func {func_name} is starting")
        result = func(*args, **kwargs)
        logging.info(f"func {func_name} is finished")
        return result
    return log_wrapper
