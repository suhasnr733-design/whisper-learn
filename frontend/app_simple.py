"""
Simplified Streamlit Frontend
"""
import streamlit as st
import requests
import json
import time

st.set_page_config(
    page_title="Whisper Learn",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎙️ Whisper Learn</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    st.subheader("Model Info")
    st.info("Whisper Small (488MB) - Optimized for 8GB RAM")
    
    # Test connection
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Connected to API")
            else:
                st.error("❌ API Error")
        except Exception as e:
            st.error(f"❌ Cannot connect: {e}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📝 Transcribe", "📊 About", "ℹ️ Help"])

with tab1:
    st.header("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'mp4', 'ogg', 'flac'],
        help="Upload lecture recording for transcription"
    )
    
    if uploaded_file:
        # Display audio player
        st.audio(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Start Transcription", type="primary", use_container_width=True):
                with st.spinner("🔄 Transcribing... This may take a moment."):
                    files = {"file": uploaded_file.getvalue()}
                    
                    try:
                        response = requests.post(
                            f"{api_url}/transcribe",
                            files=files,
                            timeout=300
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state['transcript_result'] = result
                            st.success("✅ Transcription complete!")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Display results
        if 'transcript_result' in st.session_state:
            result = st.session_state['transcript_result']
            
            st.subheader("📝 Transcript")
            transcript_text = st.text_area(
                "",
                result.get('transcript', ''),
                height=300,
                key="transcript"
            )
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Language", result.get('language', 'N/A').upper())
            with col2:
                word_count = len(result.get('transcript', '').split())
                st.metric("Words", word_count)
            with col3:
                st.metric("Status", "Success")
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Transcript (TXT)",
                    result.get('transcript', ''),
                    file_name="transcript.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    "📥 Download as JSON",
                    json.dumps(result, indent=2),
                    file_name="transcript.json",
                    mime="application/json"
                )

with tab2:
    st.markdown("""
    ## 🎯 Whisper Learn Features
    
    ### Core Features
    - **Speech-to-Text**: High accuracy transcription using OpenAI Whisper
    - **Multi-language**: Support for 100+ languages
    - **Real-time**: Live lecture processing
    
    ### Advanced Features
    - **Summarization**: AI-powered lecture summaries
    - **Slide Alignment**: Sync transcript with presentation slides
    - **Q&A System**: Ask questions about the lecture
    - **Vector Search**: Semantic search through content
    
    ### Technical Specs
    - **Model**: Whisper Small (488MB)
    - **RAM Usage**: ~2-3GB during transcription
    - **Processing**: Real-time to 2x real-time speed
    """)

with tab3:
    st.markdown("""
    ## 📖 How to Use
    
    ### Step 1: Upload Audio
    - Click "Browse files" to select your lecture recording
    - Supported formats: MP3, WAV, M4A, MP4, OGG, FLAC
    
    ### Step 2: Transcribe
    - Click "Start Transcription" button
    - Wait for processing (depends on file length)
    
    ### Step 3: Download Results
    - Copy transcript from text area
    - Download as TXT or JSON file
    
    ### Tips
    - Use high-quality audio for best results
    - Keep lectures under 1 hour for optimal performance
    - Clear background noise improves accuracy
    
    ### Troubleshooting
    - Ensure API is running on port 8000
    - Check internet connection for model downloads
    - Restart if memory issues occur
    """)

# Footer
st.markdown("---")
st.caption("Whisper Learn v1.0 | Powered by OpenAI Whisper | Optimized for 8GB RAM")