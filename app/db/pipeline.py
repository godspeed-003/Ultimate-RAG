import pixeltable as pt
from app.workflows.orchestration import workflow
from app.core.config import settings
import os

def setup_pipeline():
    """Sets up the automated Pixeltable pipeline linked to the data directory."""
    table = init_db()
    
    # Define the Pixeltable UDF for orchestration
    # This wraps our multi-stage, lazy-loading logic
    @pt.udf
    def run_intel_pipeline(doc_path: str) -> str:
        return workflow.process_document(doc_path)

    # In Pixeltable 2026, we can link a DirectoryTable to a processing UDF
    # For this MVP, we ensure the 'intelligence' column is computed automatically
    if 'intelligence' not in table.list_columns():
        logger.info("Adding automated intelligence column to Pixeltable...")
        table.add_column('intelligence', compute=run_intel_pipeline(table.document))
        
    return table

if __name__ == "__main__":
    setup_pipeline()
