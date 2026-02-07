from app.models.base_model import VisionModel
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class ChandraOCRModel(VisionModel):
    """Specific implementation for Chandra-OCR (INT8)."""
    
    def __init__(self):
        super().__init__(model_id=settings.OCR_MODEL_VERSION)

    def load(self):
        """Loads Chandra-OCR into GPU."""
        import chandra_ocr # Assumed 2026 library
        logger.info(f"Loading {self.model_id} (~2.5GB VRAM)...")
        self.model = chandra_ocr.load_model(
            model_name=self.model_id,
            precision="int8",
            device=settings.GPU_DEVICE
        )
        return self

    def run(self, input_image):
        """Processes an image for OCR and layout."""
        if not self.model:
            self.load()
        logger.info("Running Chandra OCR/Layout analysis...")
        results = self.model.analyze(input_image, features=["text", "layout", "tables", "handwriting"])
        return {
            "text": results.export_markdown(),
            "layout": results.get_layout_json(),
            "confidence": results.average_confidence()
        }

class ReasoningLLM(VisionModel):
    """Specific implementation for Qwen3-VL-8B-Instruct."""
    
    def __init__(self):
        super().__init__(model_id=settings.REASONING_LLM_VERSION)

    def load(self):
        """Loads Qwen3-VL into GPU using vLLM-style backend."""
        from vllm import LLM, SamplingParams # Assuming vLLM support in 2026
        logger.info(f"Loading {self.model_id} (~4.8GB VRAM)...")
        self.model = LLM(
            model=f"qwen/{self.model_id}",
            quantization="awq", # High performance 4-bit
            gpu_memory_utilization=0.8,
            max_model_len=4096
        )
        return self

    def run(self, prompt, image_path=None):
        """Runs reasoning/vision-llm tasks."""
        if not self.model:
            self.load()
        
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.2, max_tokens=1024)
        
        inputs = {"prompt": prompt}
        if image_path:
            inputs["multi_modal_data"] = {"image": image_path}
            
        outputs = self.model.generate([inputs], sampling_params)
        return outputs[0].outputs[0].text
