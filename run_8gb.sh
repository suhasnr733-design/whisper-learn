#!/bin/bash
# Whisper Learn - 8GB Optimized Startup Script

set -e

echo "========================================="
echo "🚀 Whisper Learn - 8GB Optimized Edition"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check available RAM
AVAILABLE_RAM=$(free -g | awk '/^Mem:/{print $7}')
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')

echo ""
echo "📊 System Information:"
echo "   Total RAM: ${TOTAL_RAM}GB"
echo "   Available RAM: ${AVAILABLE_RAM}GB"

if [ $TOTAL_RAM -lt 6 ]; then
    echo -e "${RED}⚠️ Warning: Less than 6GB total RAM. Some features may not work.${NC}"
    echo "   Recommended: Close other applications or add swap space"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p models data uploads transcripts summaries logs tmp_cache
mkdir -p models/whisper models/llm models/summarization

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements-8gb.txt -q

# Download models
echo ""
echo "🤖 Checking models..."

# Download Whisper model if not present
if [ ! -f "models/whisper/medium.pt" ] && [ ! -f "models/whisper/small.pt" ]; then
    echo "   Downloading Whisper medium model (1.5GB)..."
    python -c "
import whisper
model = whisper.load_model('medium')
print('✅ Whisper medium downloaded')
" 2>/dev/null || echo "   ⚠️ Whisper download failed, will download on first use"
fi

# Set environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=0

# Determine which services to start based on RAM
if [ $TOTAL_RAM -ge 8 ]; then
    echo -e "${GREEN}✅ Full mode: All features enabled${NC}"
    export WHISPER_MODEL=medium
    export USE_LLM=true
    export USE_SUMMARIZATION=true
elif [ $TOTAL_RAM -ge 6 ]; then
    echo -e "${YELLOW}⚠️ Limited mode: Some features disabled${NC}"
    export WHISPER_MODEL=small
    export USE_LLM=false
    export USE_SUMMARIZATION=true
else
    echo -e "${RED}⚠️ Minimal mode: Core features only${NC}"
    export WHISPER_MODEL=base
    export USE_LLM=false
    export USE_SUMMARIZATION=false
fi

# Create config file
cat > config.json << EOF
{
    "whisper_model": "${WHISPER_MODEL}",
    "use_llm": ${USE_LLM},
    "use_summarization": ${USE_SUMMARIZATION},
    "max_audio_minutes": 60,
    "cache_enabled": true,
    "memory_limit_mb": 7000
}
EOF

echo ""
echo "⚙️ Configuration saved to config.json"

# Start Redis if available
if command -v redis-server &> /dev/null; then
    echo "🔄 Starting Redis..."
    redis-server --daemonize yes --maxmemory 512mb --save "" 2>/dev/null || true
fi

# Start the API server
echo ""
echo "🎯 Starting Whisper Learn API..."
echo -e "${GREEN}📱 Access the application at: http://localhost:8000${NC}"
echo -e "${GREEN}📚 API Documentation: http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop"

# Run the API
python -c "
import sys
sys.path.insert(0, '.')

from api.main import app
import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        workers=1,
        log_level='info'
    )
" 2>&1 | tee logs/api.log