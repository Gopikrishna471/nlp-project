#streamlit run summary6.py
import streamlit as st
import fitz
import docx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import base64
from PIL import Image
import io
import time
from hello2 import NewsContentExtractor
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pretrained T5 model and tokenizer for question answering
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Set page configuration
st.set_page_config(
    page_title="Abstractive Summarizer of Long Documents",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI with background
st.markdown("""
    <style>
        /* Background Image */
        .appview-container, .css-1outpf7 {
           background-image: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)),
                              url("https://www.shutterstock.com/image-vector/newspaper-background-torn-paper-style-600nw-2261765635.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }
        /* Main Content Styling */
        .main {
            padding: 2rem;
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        /* Button Styling */
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FF2B2B;
            border-color: #FF2B2B;
        }
        /* Status Boxes */
        .status-box {
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #D4EDDA;
            color: #155724;
            border: 1px solid #C3E6CB;
        }
        .info-box {
            background-color: #E2E3E5;
            color: #383D41;
            border: 1px solid #D6D8DB;
        }
        /* Title */
        h1 {
    color: #1976D2; /* Primary blue color */
    font-size: 3rem;
    text-align: center;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #1976D2, #FF4B4B);
    -webkit-background-clip: text;
    color: transparent;
}
            h1 span {
    font-family: "emoji", sans-serif; /* Ensure emoji uses a separate font */
}

        /* Progress Bar */
        .stProgress .st-bo {
            background-color: #FF4B4B;
        }
    </style>
""", unsafe_allow_html=True)
# Define team members
team_members = [
    "M.V.N. Amruth Sai - 99220040641",
    "K. Gopi Krishna - 99220040583",
    "M. Krishna Reddy - 99220040635",
    "M. Yogi Reddy - 99220040627",
    "P. Akhil Seshu Kumar - 99220040673"


]

# Helper functions
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        return text, True
    except Exception as e:
        return str(e), False

def extract_text_from_txt(txt_file):
    try:
        return txt_file.read().decode("utf-8"), True
    except Exception as e:
        return str(e), False

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs]), True
    except Exception as e:
        return str(e), False

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_summary(text, tone="concise"):
    if tone == "concise":
        model_name = "facebook/bart-large-cnn"
        max_summary_length = 150
        min_summary_length = 50
    else:
        model_name = "facebook/bart-large-xsum"
        max_summary_length = 300
        min_summary_length = 150

    tokenizer, model = load_model(model_name)
    
    tokenized_text = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        tokenized_text, 
        max_length=max_summary_length, 
        min_length=min_summary_length, 
        length_penalty=2.0, 
        num_beams=4, 
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_key_phrases(text):
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    word_count = vectorizer.fit_transform([text]).sum(axis=0)
    phrases = sorted([(word, word_count[0, idx]) for word, idx in vectorizer.vocabulary_.items()], key=lambda x: -x[1])
    return [phrase for phrase, count in phrases[:5]]

def create_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="download-button">📥 Download {filename}</a>'
    return href

def display_pdf_preview(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    middle_page_index = len(doc) // 2
    page = doc.load_page(middle_page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the output sequence
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)

    # Decode the generated output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer



# Main Application
def main():
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("Abstractive Summarization of Long Documents")
        st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Transform long articles into concise, informative summaries</p>", unsafe_allow_html=True)
    
    # Project Description
    st.markdown("### 📄 Project Description")
    st.write("""
        This application provides an abstractive summarization of long documents, 
        aiming to transform lengthy articles into concise and informative summaries. 
        It supports PDF, DOCX, and TXT formats, making it accessible for a wide range of document types.
    """)
    
    # Explain Supported Formats
    st.markdown("### 📂 Supported File Formats")
    st.markdown("""
<div style='padding: 1rem; border-radius: 10px; background-color: #B0BEC5; color: #000000;'>
    <strong>PDF:</strong> Common document format. Ideal for magazine articles, reports, and longer texts.
</div>
<div style='padding: 1rem; border-radius: 10px; background-color: #90A4AE; color: #000000;'>
    <strong>DOCX:</strong> Word documents used for text files with formatting.
</div>
<div style='padding: 1rem; border-radius: 10px; background-color: #78909C; color: #000000;'>
    <strong>TXT:</strong> Plain text format, best for minimal content without styling.
</div>

    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    tone = st.sidebar.radio(
        "Summary Style",
        options=["Concise", "Detailed"],
        help="Choose between a brief overview or a detailed summary"
    )
    
    # Navigation Bar - Team Section
    st.sidebar.header("👥 Team Members")
    for member in team_members:
        st.sidebar.write(member)
    
    # Advanced Options
    with st.sidebar.expander("🔧 Advanced Options"):
        show_preview = st.checkbox("Show Document Preview", value=True)
        show_sentiment = st.checkbox("Show Sentiment Analysis", value=True)
        show_key_phrases = st.checkbox("Show Key Phrases", value=True)

    # Main Content Area
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )

    if uploaded_file:
        # Process the uploaded file
        with st.spinner("📑 Processing your document..."):
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            # Extract text based on file type
            if file_type == "pdf":
                if show_preview:
                    preview_image = display_pdf_preview(uploaded_file)
                    st.image(preview_image, caption="Document Preview", use_column_width=True)
                uploaded_file.seek(0)
                input_text, success = extract_text_from_pdf(uploaded_file)
            elif file_type == "txt":
                input_text, success = extract_text_from_txt(uploaded_file)
            elif file_type == "docx":
                input_text, success = extract_text_from_docx(uploaded_file)
            
            if not success:
                st.error(f"Error processing file: {input_text}")
                return

            # Create tabs for different views
            tab1, tab2 = st.tabs(["📄 Original Text", "✨ Summary"])
            
            with tab1:
                st.text_area("Original Content", input_text, height=300)
                st.markdown(f"**Word Count:** {len(input_text.split())}")
                content_extractor = NewsContentExtractor()


                st.markdown("""
    <h1 style="text-align: center; font-size: 3rem; color: #FF4B4B;">🤖 CHATBOT</h1>
""", unsafe_allow_html=True)
                main_content = content_extractor.extract_content(input_text)
                question = st.text_input("Ask a question about the content:")

                if question:
                    if input_text:
                        st.write("Generating answer...")
                        answer = generate_answer(question, main_content)  # Using input_text as the context
                        st.markdown(f"**Answer:** {answer}")
                    else:
                        st.warning("Please generate a summary first.")

            with tab2:
                if st.button("Generate Summary", key="generate"):
                    with st.spinner("🤖 Analyzing and generating summary..."):
                        # Progress bar for better user feedback
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)

                        # Extract main content
                        content_extractor = NewsContentExtractor()
                        main_content = content_extractor.extract_content(input_text)
                        
                        # Generate summary
                        summary_text = generate_summary(main_content, tone.lower())
                        
                        # Display summary in a nice box
                        st.markdown("### 📝 Summary")
                        st.markdown(f'<div class="status-box success-box">{summary_text}</div>', unsafe_allow_html=True)
                        st.markdown(f"**Summary Word Count:** {len(summary_text.split())}")
                        
                        # Additional Analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if show_sentiment:
                                sentiment = TextBlob(summary_text).sentiment
                                sentiment_label = "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral"
                                st.markdown("### 😊 Sentiment Analysis")
                                st.markdown(f'<div class="status-box info-box">Overall tone: {sentiment_label}<br>Confidence: {abs(sentiment.polarity):.2f}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            if show_key_phrases:
                                st.markdown("### 🔑 Key Phrases")
                                key_phrases = extract_key_phrases(main_content)
                                st.markdown(f'<div class="status-box info-box">{", ".join(key_phrases)}</div>', unsafe_allow_html=True)
                        
                        # Download options
                        st.markdown("### 📥 Download Options")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(create_download_link(summary_text, "summary.txt"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(create_download_link(main_content, "extracted_content.txt"), unsafe_allow_html=True)
                        

    # Ask a question
# Ask a question
                        # Ask a question after generating summary




if __name__ == "__main__":
    main()


