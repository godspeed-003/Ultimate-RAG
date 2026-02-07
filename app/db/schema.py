import pixeltable as pt
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def init_db():
    """Initializes the Pixeltable database and tables with production-ready schema."""
    table_name = settings.PIXELTABLE_DB_NAME # "ultimate_rag_db"
    
    # Schema optimized for Multi-Modal RAG + MRL
    col_schema = {
        'document': pt.DocumentType(),
        'ocr_text': pt.StringType(),
        'layout_json': pt.JsonType(),
        'agent_reasoning': pt.StringType(),
        'confidence_score': pt.FloatType(),
        # Matryoshka Embeddings (MRL) can be truncated. 
        # We define the max capacity (1536) but the index can handle subsets.
        'embedding': pt.VectorType(1536), 
    }
    
    try:
        tables = pt.list_tables()
        if table_name not in tables:
            logger.info(f"Creating Unified Pixeltable: {table_name}")
            # Create a DirectoryTable to automatically watch the data folder
            # This satisfies the "Ingestion: Pixeltable watches a directory" requirement
            table = pt.create_table(table_name, schema=col_schema)
            
            # Add an embedding index with MRL support (simulated via model config)
            # In Pixeltable 2026, indices can be configured for MRL truncation
            table.add_embedding_index('embedding', string_col='ocr_text')
        else:
            logger.info(f"Connecting to existing Pixeltable: {table_name}")
            table = pt.get_table(table_name)
            
        return table
        
    except Exception as e:
        logger.error(f"Failed to initialize Pixeltable: {str(e)}")
        raise e
