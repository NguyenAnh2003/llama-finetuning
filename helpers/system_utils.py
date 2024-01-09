import platform
from pynvml import *
import torch

# GPU utilization
def get_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB")

# get GPU services
def get_gpu_services():
    """
    The following code is used for checking CUDA service
    quantities of CUDA services prepared for training.
    Using torch.cuda
    """
    is_cuda_free = torch.cuda.is_available()
    print(f"CUDA free: {is_cuda_free}") # checking cuda service
    # get number of CUDA services
    num_cudas = torch.cuda.device_count()
    print(f"Number of devices: {num_cudas}")
    if is_cuda_free:
        for i in range(num_cudas):
            # get device props
            device = torch.device('cuda', i)
            print(f"CUDA device: {i} Name: {torch.cuda.get_device_name(i)}"
                  f"Compute capability: {torch.cuda.get_device_capability(i)}"
                  f"Total memory: {torch.cuda.get_device_properties(i).total_memory} bytes"
                  f"Props: {torch.cuda.get_device_properties(i)}")
    # get CPU
    print(f"CPU: {platform.processor()}"
          f"System: {platform.system(), platform.release()}"
          f"Py Version: {platform.python_version()}")