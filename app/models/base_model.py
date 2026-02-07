from abc import ABC, abstractmethod
from app.core.memory_manager import memory_manager
import logging

logger = logging.getLogger(__name__)

class VisionModel(ABC):
    """Base class for vision models with lazy loading capabilities."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None

    @abstractmethod
    def load(self):
        """Logic to load the model into VRAM."""
        pass

    def unload(self):
        """Unload the model and clear cache."""
        logger.info(f"Unloading model: {self.model_id}")
        self.model = None
        memory_manager.clear_gpu_cache()

    @abstractmethod
    def run(self, input_data):
        """Run inference."""
        pass
