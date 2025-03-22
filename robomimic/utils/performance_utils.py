import os
import sys
import time

import psutil


def memory_usage_psutil():
    # Return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem


def check_memory_and_time(func):
    def wrapper(*args, **kwargs):
        # Record the starting time and memory
        start_time = time.time()
        start_mem = memory_usage_psutil()

        # Run the function
        result = func(*args, **kwargs)

        # Record the ending time and memory
        end_time = time.time()
        end_mem = memory_usage_psutil()

        # Calculate the differences
        time_taken = end_time - start_time
        memory_used = end_mem - start_mem

        print(f"Function '{func.__name__}' took {time_taken:.4f} seconds to run.")
        print(f"Memory usage: {memory_used:.2f} MB")

        return result

    return wrapper

def get_size(obj, seen=None):
    """Recursively finds the size of objects in bytes."""
    if seen is None:
        seen = set()
        
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(i, seen) for i in obj)
    
    return size