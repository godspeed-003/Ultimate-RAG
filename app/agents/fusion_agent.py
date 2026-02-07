from app.models.ocr_model import ReasoningLLM
import logging

logger = logging.getLogger(__name__)

class FusionAgent:
    """Agent specialized in cross-modal validation and fusion."""
    
    def __init__(self, model: ReasoningLLM):
        self.model = model

    def validate_and_fuse(self, vision_results: str, text_results: str):
        """Cross-checks visual data against extracted text and assigns a confidence score."""
        prompt = (
            "You are a Fusion & Validation Agent. Compare the findings from a Vision Specialist and a Text Specialist.\n\n"
            "Vision Findings:\n"
            f"{vision_results}\n\n"
            "Text Findings:\n"
            f"{text_results}\n\n"
            "Calculated a final Confidence Score (0.0 to 1.0) and validation summary.\n"
            "Return EXACTLY as a JSON object with keys: 'confidence_score' (float), 'summary' (string), and 'flag' ('Green' or 'Red')."
        )
        logger.info("Fusion Agent: Cross-checking modalities...")
        response = self.model.run(prompt)
        return response
