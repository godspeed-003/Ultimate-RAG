from app.models.ocr_model import ChandraOCRModel, ReasoningLLM
from app.agents.vision_agent import VisionAgent
from app.agents.text_agent import TextAgent
from app.agents.fusion_agent import FusionAgent
from app.db.schema import init_db
from app.core.memory_manager import memory_manager
import logging

logger = logging.getLogger(__name__)

class DocumentWorkflow:
    """Manages the two-stage pipeline for document processing."""
    
    def __init__(self):
        self.table = init_db()

    def process_document(self, doc_path: str):
        """High-level orchestration of the lazy loading pipeline with real Pixeltable storage."""
        
        # --- Stage 1: OCR ---
        logger.info("=== STAGE 1: OCR PASS (Chandra-OCR) ===")
        ocr_model = ChandraOCRModel()
        ocr_results = ocr_model.run(doc_path)
        
        # Store initial record in Pixeltable
        # We insert the document path and the OCR results
        res = self.table.insert(
            document=doc_path,
            ocr_text=ocr_results['text'],
            layout_json=ocr_results['layout']
        )
        row_id = res.inserted_ids[0] # Get the ID for the inserted row
        
        # Clear VRAM for the reasoning model
        ocr_model.unload()
        
        # --- Stage 2: Reasoning & Multi-Agent Logic ---
        logger.info("=== STAGE 2: REASONING PASS (Qwen3-VL) ===")
        reasoning_model = ReasoningLLM()
        
        # Initialize Agents with the reasoning model
        vision_agent = VisionAgent(reasoning_model)
        text_agent = TextAgent(reasoning_model)
        fusion_agent = FusionAgent(reasoning_model)
        
        # Sequential multi-agent reasoning
        vision_info = vision_agent.analyze_visuals(doc_path)
        text_info = text_agent.extract_details(ocr_results['text'])
        
        # Final Cross-Modal Fusion
        final_validation_raw = fusion_agent.validate_and_fuse(vision_info, text_info)
        
        # --- Innovation: Self-Correction Loop ---
        import json
        try:
            val_data = json.loads(final_validation_raw)
            confidence_score = val_data.get('confidence_score', 0.0)
        except:
            confidence_score = 0.5 # Fallback
            val_data = {"summary": "Error parsing agent output", "flag": "Red"}

        if confidence_score < 0.7:
            logger.warning(f"Confidence score {confidence_score} is below 0.7. Triggering Thinking Mode...")
            # Thinking Mode: A more detailed, step-by-step reasoning prompt
            thinking_prompt = (
                "CRITICAL RE-ANALYSIS: The previous fusion attempt had low confidence. "
                "Switching to deep Thinking Mode. Please analyze the discrepancies between the visual stamps "
                "and the extracted text again, step-by-step. "
                "Visuals: {vision_info}\nText: {text_info}"
            )
            final_validation_raw = reasoning_model.run(thinking_prompt, image_path=doc_path)
            # Re-parse or just use the new summary
            try:
                # Assuming thinking mode returns the same JSON structure
                val_data = json.loads(final_validation_raw)
                confidence_score = val_data.get('confidence_score', confidence_score)
            except:
                val_data = {"summary": final_validation_raw, "flag": "Yellow", "confidence_score": confidence_score}

        # Update Pixeltable with specialized agent intelligence
        self.table.update(
            {
                'agent_reasoning': val_data.get('summary', final_validation_raw),
                'confidence_score': float(confidence_score)
            },
            where=self.table.id == row_id
        )
        
        # Clear VRAM
        reasoning_model.unload()
        
        logger.info("Pipeline Complete.")
        return final_validation_raw

workflow = DocumentWorkflow()
