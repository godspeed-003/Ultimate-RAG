from app.models.ocr_model import ReasoningLLM
import logging

logger = logging.getLogger(__name__)

class VisionAgent:
    """Agent specialized in visual reasoning and landmark detection."""
    
    def __init__(self, model: ReasoningLLM):
        self.model = model

    def analyze_visuals(self, image_path: str):
        """Identifies visual landmarks (signatures, site plans, stamp presence)."""
        prompt = (
            "Analyze the provided document image. Specifically look for:\n"
            "1. Presence of official stamps.\n"
            "2. Presence of signatures.\n"
            "3. If there are any site plans or technical diagrams.\n"
            "Return a JSON-formatted summary of your findings."
        )
        logger.info("Vision Agent: Analyzing visual landmarks...")
        response = self.model.run(prompt, image_path=image_path)
        return response
