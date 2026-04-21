cat > README.md << 'EOF'
# 🎙️ Whisper Learn - Intelligent Lecture-to-Text Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Real-time Transcription** - Live lecture processing with Whisper
- **Multi-language Support** - Auto-detect + translate 100+ languages
- **Slide Integration** - Align transcripts with PDF/PPT slides
- **Smart Summarization** - AI-powered lecture summaries
- **Vector Search** - RAG-based Q&A on lecture content
- **Meeting Bot** - Auto-join Zoom/Teams calls
- **Optimized for 8GB RAM** - Efficient memory management

## 📋 Requirements

- Python 3.9+
- 8GB RAM (6GB usable)
- 10GB free storage
- Internet connection (for first-time setup)

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/whisper-learn-8gb.git
cd whisper-learn-8gb

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-8gb.txt

# Download models
python -c "import whisper; whisper.load_model('small')"