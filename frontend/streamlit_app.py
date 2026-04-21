"""
Streamlit frontend for Whisper Learn
"""
import streamlit as st
import requests
import json
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="Whisper Learn",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎙️ Whisper Learn</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center'>Intelligent Lecture-to-Text Pipeline</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    st.subheader("Transcription Options")
    language = st.selectbox("Language", ["auto", "en", "es", "fr", "de", "zh", "ja"])
    task = st.selectbox("Task", ["transcribe", "translate"])
    
    st.subheader("Processing Options")
    summarize = st.checkbox("Generate Summary", value=True)
    summary_style = st.selectbox("Summary Style", ["hybrid", "extractive", "abstractive"])
    
    st.subheader("System Status")
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Not Reachable")
    
    try:
        mem_response = requests.get(f"{api_url}/memory", timeout=2)
        if mem_response.status_code == 200:
            mem_data = mem_response.json()
            st.metric("Available RAM", f"{mem_data.get('available_gb', 0):.1f} GB")
            st.metric("RAM Usage", f"{mem_data.get('percent_used', 0)}%")
    except:
        pass

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📝 Transcribe", "🎤 Live Lecture", "💬 Q&A", "📊 History"])

with tab1:
    st.header("Upload Audio/Video")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['mp3', 'mp4', 'wav', 'm4a', 'ogg'],
        help="Upload lecture recording"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file:
            st.audio(uploaded_file)
    
    with col2:
        if st.button("🚀 Start Transcription", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner("Processing..."):
                    files = {"file": uploaded_file.getvalue()}
                    params = {
                        "language": None if language == "auto" else language,
                        "task": task,
                        "summarize": summarize,
                        "summary_style": summary_style
                    }
                    
                    response = requests.post(
                        f"{api_url}/transcribe",
                        files=files,
                        params=params
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['last_transcript'] = result
                        st.success("✅ Transcription complete!")
                    else:
                        st.error(f"Error: {response.text}")
    
    if 'last_transcript' in st.session_state:
        result = st.session_state['last_transcript']
        
        st.subheader("📝 Transcript")
        st.text_area("", result['transcript'], height=300)
        
        if 'summary' in result:
            st.subheader("📚 Summary")
            st.markdown(result['summary']['summary'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Words", len(result['transcript'].split()))
            with col2:
                st.metric("Duration", f"{result.get('duration_seconds', 0):.0f}s")
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Transcript",
                result['transcript'],
                file_name="transcript.txt",
                mime="text/plain"
            )
        with col2:
            if 'summary' in result:
                st.download_button(
                    "📥 Download Summary",
                    result['summary']['summary'],
                    file_name="summary.txt",
                    mime="text/plain"
                )

with tab2:
    st.header("🎤 Live Lecture Recording")
    
    st.info("🎙️ Click 'Start Recording' to begin capturing your lecture in real-time")
    
    duration = st.slider("Recording Duration (minutes)", 1, 60, 15)
    
    if st.button("🔴 Start Recording", type="primary"):
        st.warning("Recording in progress... (simulated for demo)")
        progress_bar = st.progress(0)
        
        for i in range(duration):
            time.sleep(0.05)  # Simulated progress
            progress_bar.progress((i + 1) / duration)
        
        st.success("✅ Recording complete! Processing transcription...")
        st.info("In production, this would stream audio to Whisper in real-time")

with tab3:
    st.header("💬 Ask Questions About Your Lecture")
    
    if 'last_transcript' not in st.session_state:
        st.info("Please transcribe a lecture first to enable Q&A")
    else:
        transcript = st.session_state['last_transcript']['transcript']
        
        question = st.text_input("Ask a question about the lecture:")
        
        if question and st.button("Ask"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{api_url}/ask",
                    json={
                        "question": question,
                        "context": transcript[:5000]  # Limit context
                    }
                )
                
                if response.status_code == 200:
                    answer = response.json()
                    st.markdown(f"**Answer:** {answer['answer']}")
                    st.caption(f"Confidence: {answer.get('confidence', 0):.2f}")
                else:
                    st.error("Error getting answer")

with tab4:
    st.header("📊 Session History")
    
    st.info("Previous transcriptions will appear here")
    
    # Placeholder for history
    st.json({
        "total_transcriptions": 0,
        "total_minutes": 0,
        "last_session": None
    })

# Footer
st.markdown("---")
st.caption("Whisper Learn - Powered by OpenAI Whisper | Optimized for 8GB RAM")