import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import fitz  # PyMuPDF
import io
import time
from datetime import datetime

# --------------------------
# Load Model & Tokenizer
# --------------------------
@st.cache_resource
def load_model():
    MODEL_PATH = "C:\\Users\\jayav\\Downloads\\Save\\Projects\\Plagarism-Detection\\model"  # Update with actual path
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

tokenizer, model, device = load_model()

# --------------------------
# Utility Functions
# --------------------------
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file with error handling."""
    try:
        pdf_bytes = uploaded_file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def predict_similarity(text1, text2, max_len=192):
    """Runs model on two texts and returns prediction + probability."""
    if not model or not tokenizer:
        return None, None
        
    try:
        inputs = tokenizer(
            text1, text2,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = int(torch.argmax(outputs.logits, dim=1).item())

        return pred, probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --------------------------
# Streamlit Configuration
# --------------------------
st.set_page_config(
    page_title="Repellica | AI-Powered Plagiarism Detection",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------
# Advanced CSS Styling
# --------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #000000 0%, #1a1a2e 50%, #16213e 100%);
        padding: 40px 20px;  /* Reduced padding */
        text-align: center;
        border-radius: 0 0 30px 30px;  /* Smaller border radius */
        margin-bottom: 20px;  /* Reduced margin */
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #ffffff 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #b0b0b0;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto 30px;
        line-height: 1.6;
    }
    
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 40px;
        flex-wrap: wrap;
    }
    
    .stat-item {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        min-width: 150px;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #b0b0b0;
        margin-top: 5px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;  /* Reduced padding */
        margin: 15px 0;  /* Reduced margin */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 15px;  /* Reduced gap */
        margin: 20px 0;  /* Reduced margin */
    }
    
    .feature-item {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .feature-item:hover {
        transform: scale(1.05);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        display: block;
    }
    
    .feature-title {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        color: #b0b0b0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #ffffff;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0099cc 0%, #00d4ff 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
    }
    
    .upload-area {
        border: 2px dashed rgba(0, 212, 255, 0.5);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background: rgba(0, 212, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #00d4ff;
        background: rgba(0, 212, 255, 0.1);
    }
    
    .result-success {
        background: linear-gradient(135deg, rgba(0, 255, 0, 0.1) 0%, rgba(0, 200, 0, 0.1) 100%);
        border: 1px solid rgba(0, 255, 0, 0.3);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        color: #ffffff;
    }
    
    .result-warning {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 0, 0, 0.1) 100%);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        color: #ffffff;
    }
    
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #ff6b6b);
    }
    
    .footer {
        text-align: center;
        padding: 30px 20px;  /* Reduced padding */
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 40px;  /* Reduced margin */
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: #00d4ff;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .footer-link:hover {
        color: #ffffff;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 40px;
    }
    
    .spinner {
        width: 50px;
        height: 50px;
        border: 3px solid rgba(0, 212, 255, 0.3);
        border-top: 3px solid #00d4ff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .confidence-meter {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .section-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin: 20px 0 15px;  /* Reduced margins */
    }
    
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .stats-container {
            gap: 15px;
        }
        .stat-item {
            min-width: 120px;
            padding: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Hero Section
# --------------------------
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">Repellica</div>
        <div class="hero-subtitle">
            Next-generation AI-powered plagiarism detection using advanced natural language processing. 
            Built with cutting-edge transformer models for unparalleled accuracy.
        </div>
        <div class="stats-container">
            <div class="stat-item">
                <span class="stat-number">95.2%</span>
                <span class="stat-label">Accuracy Rate</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">135M</span>
                <span class="stat-label">Parameters</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">Open</span>
                <span class="stat-label">Source</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Features Section
# --------------------------
st.markdown('<h2 class="section-title">🌟 Why Choose Repellica?</h2>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="feature-grid">
        <div class="feature-item">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">AI-Powered Detection</div>
            <div class="feature-desc">Fine-tuned SmolLM model trained on MIT Plagiarism Detection Dataset for superior accuracy</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Lightning Fast</div>
            <div class="feature-desc">Process documents in seconds with optimized GPU acceleration and efficient algorithms</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">🔒</span>
            <div class="feature-title">Privacy First</div>
            <div class="feature-desc">Your documents are processed locally and never stored or transmitted to external servers</div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">High Precision</div>
            <div class="feature-desc">Advanced semantic analysis detects even sophisticated paraphrasing and restructuring attempts</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Main Application
# --------------------------
st.markdown('<h2 class="section-title">📄 Document Analysis</h2>', unsafe_allow_html=True)

# File upload section
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📄 First Document")
    pdf_file1 = st.file_uploader(
        "Choose the first PDF file", 
        type="pdf", 
        key="pdf1",
        help="Upload the first document for comparison (max 200MB)"
    )
    if pdf_file1:
        st.success(f"✅ {pdf_file1.name} uploaded successfully!")
        st.info(f"File size: {pdf_file1.size / (1024*1024):.1f} MB")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Second Document")
    pdf_file2 = st.file_uploader(
        "Choose the second PDF file", 
        type="pdf", 
        key="pdf2",
        help="Upload the second document for comparison (max 200MB)"
    )
    if pdf_file2:
        st.success(f"✅ {pdf_file2.name} uploaded successfully!")
        st.info(f"File size: {pdf_file2.size / (1024*1024):.1f} MB")
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis section
if pdf_file1 and pdf_file2:
    # Extract text with progress indication
    with st.spinner("🔍 Extracting and preprocessing document content..."):
        progress_bar = st.progress(0)
        
        # Extract first document
        progress_bar.progress(25)
        text1 = extract_text_from_pdf(pdf_file1)
        
        # Extract second document  
        progress_bar.progress(75)
        text2 = extract_text_from_pdf(pdf_file2)
        
        progress_bar.progress(100)
        time.sleep(0.5)  # Brief pause for UX
        progress_bar.empty()

    if text1 and text2:
        # Document preview section
        st.markdown('<h3 class="section-title">📜 Document Previews</h3>', unsafe_allow_html=True)
        
        prev_col1, prev_col2 = st.columns(2)
        
        with prev_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### First Document Preview")
            preview_text1 = text1[:1000] + ("..." if len(text1) > 1000 else "")
            st.text_area(
                "Preview", 
                preview_text1, 
                height=200, 
                key="preview1",
                disabled=True
            )
            st.caption(f"Total characters: {len(text1):,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with prev_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Second Document Preview")
            preview_text2 = text2[:1000] + ("..." if len(text2) > 1000 else "")
            st.text_area(
                "Preview", 
                preview_text2, 
                height=200, 
                key="preview2",
                disabled=True
            )
            st.caption(f"Total characters: {len(text2):,}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Analysis button
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        
        if st.button("🚀 Start Plagiarism Analysis", type="primary"):
            start_time = time.time()
            
            with st.spinner("🔎 Analyzing documents with AI model..."):
                # Create progress bar for analysis
                analysis_progress = st.progress(0)
                
                for i in range(10):
                    time.sleep(0.1)
                    analysis_progress.progress((i + 1) * 10)
                
                pred, probs = predict_similarity(text1, text2)
                analysis_progress.empty()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if pred is not None and probs is not None:
                # Results section
                st.markdown('<h3 class="section-title">📊 Analysis Results</h3>', unsafe_allow_html=True)
                
                # Result display
                if pred == 1:
                    st.markdown(
                        f"""
                        <div class="result-warning">
                            <div style="text-align: center;">
                                <h2 style="margin: 0; color: #ff6b6b;">⚠️ Plagiarism Detected</h2>
                                <p style="font-size: 1.2rem; margin: 15px 0;">Significant similarity found between documents</p>
                                <div style="font-size: 2rem; font-weight: bold; color: #ff6b6b;">
                                    {float(probs[1]*100):.1f}% Confidence
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-success">
                            <div style="text-align: center;">
                                <h2 style="margin: 0; color: #00ff00;">✅ No Plagiarism Detected</h2>
                                <p style="font-size: 1.2rem; margin: 15px 0;">Documents appear to be original</p>
                                <div style="font-size: 2rem; font-weight: bold; color: #00ff00;">
                                    {float(probs[0]*100):.1f}% Confidence
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Confidence meter
                st.markdown('<div class="confidence-meter">', unsafe_allow_html=True)
                st.markdown("#### Confidence Distribution")
                
                confidence = float(probs[pred] * 100)  # Convert to Python float
                st.progress(confidence / 100)
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Original Probability", 
                        f"{float(probs[0]*100):.1f}%",
                        delta=f"{float((probs[0] - 0.5)*100):.1f}%" if float(probs[0]) != 0.5 else "0%"
                    )
                
                with col2:
                    st.metric(
                        "Plagiarism Probability", 
                        f"{float(probs[1]*100):.1f}%",
                        delta=f"{float((probs[1] - 0.5)*100):.1f}%" if float(probs[1]) != 0.5 else "0%"
                    )
                
                with col3:
                    st.metric(
                        "Processing Time",
                        f"{processing_time:.2f}s",
                        delta=None
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Timestamp
                st.caption(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            else:
                st.error("❌ Analysis failed. Please check your model configuration and try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("❌ Failed to extract text from one or both PDF files. Please ensure the files are valid PDFs with readable text.")

else:
    # Upload prompt
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 60px 30px;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📄</div>
            <h3 style="color: #ffffff; margin-bottom: 15px;">Upload Two PDF Documents</h3>
            <p style="color: #b0b0b0; font-size: 1.1rem; margin-bottom: 0;">
                Select two PDF files above to begin the plagiarism analysis. 
                Our AI will compare the documents and provide detailed similarity metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------
# Footer
# --------------------------
st.markdown(
    """
    <div class="footer">
        <div class="footer-content">
            <h3 style="color: #ffffff; margin-bottom: 20px;">🦁 Repellica</h3>
            <p style="color: #b0b0b0; margin-bottom: 20px;">
                Empowering academic integrity and content authenticity through advanced AI technology.
                Built with passion for open-source innovation.
            </p>
            <div class="footer-links">
                <a href="https://github.com/jayavanth18" class="footer-link" target="_blank">👨‍💻 Developer</a>
                <a href="https://github.com/plagguard" class="footer-link" target="_blank">🔗 GitHub</a>
                <a href="mailto:jayavanth18@gmail.com" class="footer-link">📧 Contact</a>
            </div>
            <p style="color: #666; font-size: 0.9rem; margin-top: 30px;">
                © 2025 Repellica. Made by Jai. Built with ❤️ using Streamlit & PyTorch.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)