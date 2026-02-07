import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "Ultimate RAG"
    
    # Model Configs
    # Memory Footprint: ~2.5 GB
    OCR_MODEL_VERSION: str = "Chandra-OCR-INT8"
    # Memory Footprint: ~4.8 GB
    REASONING_LLM_VERSION: str = "Qwen3-VL-8B-Instruct-Q4_K_M"
    
    # Hardware Constraints
    VRAM_LIMIT_GB: float = 6.0 #since I only have 6gb vram in my laptop
    GPU_DEVICE: str = "cuda:0"
    
    # Paths
    DATA_DIR: str = os.path.abspath("data")
    STORAGE_DIR: str = os.path.abspath("storage")
    
    # Pixeltable Config
    PIXELTABLE_DB_NAME: str = "ultimate_rag_db"

    class Config:
        env_file = ".env"

settings = Settings()
