from app.models.ocr_model import ReasoningLLM
import logging

logger = logging.getLogger(__name__)

class TextAgent:
    """Agent specialized in reasoning over extracted text/markdown."""
    
    def __init__(self, model: ReasoningLLM):
        self.model = model

    def extract_details(self, markdown_text: str):
        """Extracts key project details (Dates, Totals, Clauses) from OCR results."""
        prompt = (
            "Review the following extracted document text (Markdown format):\n\n"
            f"{markdown_text}\n\n"
            "Extract the following details as JSON:\n"
            "1. Project Name / Title.\n"
            "2. Total Amount or Budget (if mentioned).\n"
            "3. Key Dates (Submission, Deadlines).\n"
            "4. Main Stakeholders."
        )
        logger.info("Text Agent: Extracting structured details...")
        response = self.model.run(prompt)
        return response
