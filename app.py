import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types

# Load environment variables strictly from keys.env
load_dotenv("keys.env")

# Page Configuration & Styling
st.set_page_config(
    page_title="DocuPilot AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        padding-top: 1.5rem;
    }
    
    /* Sidebar Logo Styling */
    [data-testid="stSidebar"] img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 0.5rem;
    }
    
    /* Custom Card Containers */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 85%;
        font-weight: 600;
        border-radius: 6px;
        color: #0f5132;
        background-color: #d1e7dd;
    }
    
    /* Header Enhancements */
    .app-title {
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    
    /* Hide Default Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    extracted_text = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)
    return "\n".join(extracted_text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts text from an uploaded DOCX file."""
    doc = Document(io.BytesIO(file_bytes))
    extracted_text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text]
    return "\n".join(extracted_text)

def analyze_document_with_gemini(document_text: str, api_key: str) -> str:
    """Sends extracted document text to Gemini model for actionable insights."""
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert business document analyst.
    Examine the following document text carefully and produce a clear, structured analysis using Markdown:

    ### Executive Summary
    A concise 2-3 sentence overview of the document's core intent.

    ### Key Actionable Tasks & Next Steps
    - Clear, itemized task recommendations or follow-up actions.

    ### Key Metrics & Financial Figures
    - Tabular or bulleted list of essential numbers, dates, costs, or deadlines.

    ### Risk Analysis & Observations
    - Potential red flags, missing details, or critical observations.

    Document Text:
    {document_text}
    """
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        )
    )
    return response.text

# Sidebar Controls
with st.sidebar:
    st.image("logo.png", width=200)
    st.title("DocuPilot AI")
    st.caption("v1.0 • Powered by Gemini 2.0")
    
    st.divider()
    
    # API Key Validation Check
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and gemini_api_key.strip() != "your_actual_gemini_api_key_here":
        st.markdown('<span class="status-badge">🟢 API Key Connected</span>', unsafe_allow_html=True)
    else:
        st.error("🔴 API Key Missing in `keys.env`")
        
    st.divider()
    
    st.markdown("**Supported Formats**")
    st.markdown("- PDF Documents (`.pdf`)")
    st.markdown("- Word Documents (`.docx`)")
    
    st.divider()
    st.info("💡 **Tip:** Ensure uploaded files contain selectable text. Scanned images require OCR preprocessing.")

# Main Interface Header
st.markdown('<h1 class="app-title">DocuPilot AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Transform raw business documents into structured, actionable insights instantly.</p>', unsafe_allow_html=True)

if not gemini_api_key or gemini_api_key.strip() == "your_actual_gemini_api_key_here":
    st.error("⚠️ `GEMINI_API_KEY` is not properly configured in `keys.env`. Please add your key and restart Streamlit.")
    st.stop()

# Document Uploader Card
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Business Document", 
        type=["pdf", "docx"],
        help="Select a PDF or DOCX file to analyze."
    )

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    # Quick File Stats Display
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**Filename:** `{uploaded_file.name}`")
        st.markdown(f"**Size:** `{len(file_bytes) / 1024:.1f} KB`")
        st.markdown(f"**Type:** `{file_extension.upper()}`")
        st.markdown('</div>', unsafe_allow_html=True)

    extracted_text = ""
    try:
        if file_extension == "pdf":
            extracted_text = extract_text_from_pdf(file_bytes)
        elif file_extension == "docx":
            extracted_text = extract_text_from_docx(file_bytes)
    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        st.stop()
        
    if not extracted_text.strip():
        st.warning("⚠️ Could not extract readable text from this file. If it is a scanned image/PDF, standard parsing cannot read it without OCR.")
        st.stop()

    # Organized Output Tabs
    st.write("---")
    tab1, tab2 = st.tabs(["🚀 Actionable Insights", "📄 Extracted Raw Text"])

    with tab2:
        st.caption("Parsed plain text from uploaded document:")
        st.text_area("Raw Text", extracted_text, height=350, label_visibility="collapsed")

    with tab1:
        st.markdown("##### Ready to process document text through Gemini AI")
        if st.button("Generate Action Insights", type="primary", use_container_width=True):
            with st.spinner("Analyzing document structure, extracting key figures, and identifying action items..."):
                try:
                    analysis_result = analyze_document_with_gemini(extracted_text, gemini_api_key)
                    st.success("Analysis Complete!")
                    st.markdown("---")
                    st.markdown(analysis_result)
                except Exception as e:
                    st.error(f"API Error: {str(e)}")

else:
    # Empty State Dashboard Placeholder
    st.info("👆 Upload a document above to get started.")