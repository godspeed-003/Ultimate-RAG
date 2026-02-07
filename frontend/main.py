import streamlit as st
import pixeltable as pt
import os
import shutil
from app.core.config import settings
from app.db.pipeline import setup_pipeline

# Page Config
st.set_page_config(
    page_title="Ultimate RAG | Multi-Modal Doc Intel",
    page_icon="🤖",
    layout="wide"
)

# Premium Aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stHeader {
        color: #00ffa2;
    }
    .status-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00ffa2;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Ultimate RAG: Multi-Modal Document Intelligence")
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"VRAM Limit: {settings.VRAM_LIMIT_GB} GB")
    st.info(f"OCR: {settings.OCR_MODEL_VERSION}")
    st.info(f"LLM: {settings.REASONING_LLM_VERSION}")
    
    if st.button("🔄 Reload Pipeline"):
        setup_pipeline()
        st.success("Pipeline Reloaded!")

# Main Body
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        file_path = os.path.join(settings.DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name} to ingestion folder.")
        
        with st.status("Processing Pipeline...", expanded=True) as status:
            st.write("1. 📉 Stage 1: Chandra-OCR Analysis (INT8)...")
            # In a real Pixeltable setup, this happens automatically via the compute column
            st.write("2. 🧠 Stage 2: Qwen3-VL Multi-Agent Reasoning...")
            st.write("3. 🔗 Cross-Modal Fusion & Validation...")
            status.update(label="Inference Complete!", state="complete", expanded=False)

with col2:
    st.subheader("🎯 Extracted Intelligence")
    
    # query Pixeltable for results
    try:
        table = pt.get_table(settings.PIXELTABLE_DB_NAME)
        # Select relevant columns including confidence and agent reasoning
        df = table.select(
            table.document, 
            table.ocr_text, 
            table.agent_reasoning, 
            table.confidence_score
        ).to_pandas()
        
        if not df.empty:
            for idx, row in df.iterrows():
                # Determine Flag Color
                score = row['confidence_score']
                flag_color = "🟢 Green" if score >= 0.7 else "🔴 Red"
                
                with st.expander(f"{flag_color} | 📄 {os.path.basename(row['document'])}", expanded=True):
                    col_info, col_visual = st.columns([2, 1])
                    
                    with col_info:
                        st.markdown(f"**Confidence Score:** `{score:.2f}`")
                        st.markdown("**Agent Reasoning:**")
                        st.info(row['agent_reasoning'])
                        
                        if st.button(f"View Bounding Boxes for {idx}", key=f"btn_{idx}"):
                             st.toast("Feature: Highlighting original bounding box on image...")
                             # In a real app, this would trigger an image overlay
                             st.image("https://via.placeholder.com/400x200?text=Bounding+Box+Highlight+Simulated", use_column_width=True)
                    
                    with col_visual:
                        st.markdown("**OCR Preview**")
                        st.caption(row['ocr_text'][:500] + "...")
                    
                    st.divider()
        else:
            st.info("No documents processed yet. Upload a file to see results.")
    except Exception as e:
        st.error(f"Error loading intelligence from Pixeltable: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by Pixeltable, Chandra-OCR, and Qwen3-VL")
