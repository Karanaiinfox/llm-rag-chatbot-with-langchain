import logging
import logging.config
import psutil
#import torch
import os


# Get the logger
logger = logging.getLogger()

def configure_logging(log_config_path):
    try:
        if os.path.exists(log_config_path):
            logging.config.fileConfig(log_config_path)
        else:
            logging.basicConfig(level=logging.INFO)
        return logging.getLogger()
    except Exception as e:
        logger.error(f"Error configuring logging: {e}")
        raise


def get_memory_usage():
    """
    Function to track and print CPU and GPU memory usage
    """
    try: 
        # Get CPU memory usage
        memory = psutil.virtual_memory()
        cpu_memory = memory.percent  # Memory usage percentage of the system

        # Get GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # in MB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # in MB
            gpu_memory = {
                "allocated_memory": gpu_memory_allocated,
                "reserved_memory": gpu_memory_reserved,
            }
        else:
            gpu_memory = None
        return cpu_memory, gpu_memory
    
    except Exception as e:
        logger.error(f"Error monitoring memory usage: {e}")
        return None