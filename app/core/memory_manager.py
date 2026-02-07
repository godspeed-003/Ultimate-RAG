import torch
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages GPU memory and model lifecycle to prevent OOM on 6GB cards."""
    
    @staticmethod
    def clear_gpu_cache():
        """Forcefully clears the GPU cache and triggers Python GC."""
        logger.info("Clearing GPU cache and triggering garbage collection...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logger.info("GPU cache cleared.")

    @staticmethod
    def get_vram_usage():
        """Returns the current VRAM usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    @staticmethod
    def unload_model(model):
        """Unloads a specific model from memory."""
        if model is not None:
            del model
            MemoryManager.clear_gpu_cache()

memory_manager = MemoryManager()
