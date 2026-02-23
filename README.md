# WsprFlowPy

An open-source, lightweight, and high-performance dictation tool with AI-powered formatting - an unofficial Python remake of [wisprflow](https://wisprflow.ai/).

## System Requirements

- **OS**: Windows 10/11 (uses Windows APIs for window detection and keyboard input)
- **GPU**: NVIDIA GPU with CUDA support (recommended for best performance)
  - CPU mode available but significantly slower
- **RAM**: 4GB+ (8GB+ recommended for larger Whisper models)
- **Microphone**: Any input device (USB microphones like HyperX QuadCast work great)

## Prerequisites

### 1. Python 3.9+

Download and install Python from [python.org](https://www.python.org/downloads/)

### 2. NVIDIA GPU Setup (For GPU Acceleration)

#### Install NVIDIA Driver
- Download the latest driver from [NVIDIA's website](https://www.nvidia.com/Download/index.aspx)
- Select your GPU model and install

#### Install CUDA Toolkit 12.4

1. Download [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
2. Run the installer and select:
   - CUDA Toolkit
   - CUDA Development
   - CUDA Runtime
3. Default installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`

#### Install cuDNN 9.x

1. Download [cuDNN 9.17 for CUDA 12.x](https://developer.nvidia.com/cudnn)
2. Extract the archive
3. Copy files to CUDA installation:
   ```
   cuDNN/bin/*.dll → C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9\
   cuDNN/include/*.h → C:\Program Files\NVIDIA\CUDNN\v9.17\include\
   cuDNN/lib/*.lib → C:\Program Files\NVIDIA\CUDNN\v9.17\lib\
   ```

#### Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

**Note**: If you don't have an NVIDIA GPU, you can still use CPU mode by changing `WHISPER_DEVICE = "cpu"` in `wsprv2.py` (line 30).

### 3. FFmpeg (Optional but Recommended)

While not strictly required for this project, FFmpeg can improve audio compatibility:

1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add to PATH: `C:\ffmpeg\bin`

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wsprflowpy.git
cd wsprflowpy
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```env
   # Optional: Hugging Face token for downloading models
   HF_TOKEN=your_huggingface_token_here

   # Required: OpenRouter API key for Claude formatting
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

#### Getting API Keys

- **OpenRouter**: Sign up at [openrouter.ai](https://openrouter.ai/) and create an API key
  - Used for AI-powered transcript formatting with Claude
  - Pay-per-use pricing (very affordable for personal use)
- **Hugging Face** (optional): Sign up at [huggingface.co](https://huggingface.co/)
  - Only needed if you encounter model download issues

### 5. Verify CUDA Paths

The application automatically adds CUDA to PATH (lines 22-23 in `wsprv2.py`). Verify these paths match your installation:

```python
os.environ['PATH'] = r'C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9' + os.pathsep + os.environ['PATH']
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin' + os.pathsep + os.environ['PATH']
```

Adjust if your CUDA/cuDNN is installed elsewhere.

## Usage

### First Run

1. Start the application:
   ```bash
   python wsprv2.py
   ```

2. **Microphone Selection**: On first run, you'll be prompted to select your microphone:
   ```
   Available input devices:
   [0] Microsoft Sound Mapper - Input
   [1] HyperX QuadCast S (2- HyperX)
   [2] Microphone Array (Realtek)
   ...
   Select input device index: 1
   ```
   - Your selection is saved to `mic_config.json` and remembered for future runs
   - The app will auto-suggest preferred mics (HyperX, QuadCast, etc.)

3. Wait for initialization:
   ```
   --- Initializing Clients ---
   Loading Whisper model...
   Whisper model loaded in 2.34s.
   OpenRouter client initialized.
   ```

4. You're ready! The status badge will appear at the bottom center of your screen.

### Controls

- **Ctrl + Alt (Hold)**: Start recording for dictation
  - Hold both keys and speak
  - Release to stop recording
  - Transcribed and formatted text is automatically pasted

### Status Badge

The minimal UI badge shows current status:
- **Thin line**: Idle, ready to record
- **Red waveform**: Recording in progress
- **Pulsing dots**: Processing audio

## Configuration

### Model Settings

Edit `wsprv2.py` to customize:

```python
# Whisper STT Configuration (line 29-32)
WHISPER_MODEL_NAME = "small.en"      # tiny.en, base.en, small.en, medium.en, large-v3
WHISPER_DEVICE = "cuda"              # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16"     # float16 (GPU), int8 (CPU/low-end GPU)
WHISPER_BEAM_SIZE = 5                # Higher = more accurate but slower

# LLM for formatting (line 35)
MODEL_FORMATTER = "anthropic/claude-haiku-4.5"  # Fast and cost-effective
```

#### Model Size vs Performance

| Model | Size | Speed (GPU) | Accuracy | Use Case |
|-------|------|-------------|----------|----------|
| tiny.en | 39MB | Fastest | Good | Quick notes, drafts |
| base.en | 74MB | Very fast | Better | General use |
| small.en | 244MB | Fast | Great | **Recommended** |
| medium.en | 769MB | Moderate | Excellent | High accuracy needed |
| large-v3 | 1.5GB | Slower | Best | Maximum accuracy |

### Microphone Settings

- **Saved config**: `mic_config.json` stores your microphone selection
- **Reset config**: Delete `mic_config.json` to reselect your microphone
- **Preferred mics**: Edit `MIC_PREFERRED_KEYWORDS` in `wsprv2.py` (line 72)

### Sound Effects

Place custom sound files in the `sounds/` directory:
- `dictation-start.wav`: Recording started
- `dictation-stop.wav`: Recording stopped
- `Notification.wav`: Errors or short recordings

## Troubleshooting

### "CUDA not found" or GPU errors

1. Verify CUDA installation: `nvcc --version`
2. Check cuDNN files are in the correct location
3. As a fallback, switch to CPU mode:
   ```python
   WHISPER_DEVICE = "cpu"
   WHISPER_COMPUTE_TYPE = "int8"
   ```

### Microphone not working

1. Check Windows sound settings - ensure your mic is set as default recording device
2. Delete `mic_config.json` and restart to reselect
3. Try selecting "system default" option when prompted

### Transcription quality issues

1. Use a better microphone (USB mics recommended)
2. Speak clearly and reduce background noise
3. Try a larger Whisper model: `medium.en` or `large-v3`
4. Increase beam size: `WHISPER_BEAM_SIZE = 10`

### Slow transcription

1. Ensure you're using CUDA (`WHISPER_DEVICE = "cuda"`)
2. Use a smaller model: `tiny.en` or `base.en`
3. Check GPU usage with `nvidia-smi` during transcription

### Import errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# If specific package fails, install individually:
pip install faster-whisper --upgrade
```

### Keys not detected

- Ensure the app has focus (click the status badge)
- Try running as Administrator
- Check if another app is intercepting the hotkeys

## Building an Executable (Optional)

To create a standalone `.exe`:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --add-data "sounds;sounds" wsprv2.py
```

The executable will be in the `dist/` folder.

## Project Structure

```
wsprflowpy/
├── wsprv2.py              # Main application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .env.example          # Example environment file
├── mic_config.json       # Microphone settings (auto-generated)
├── sounds/               # Sound effect files
│   ├── dictation-start.wav
│   ├── dictation-stop.wav
│   └── Notification.wav
└── README.md            # This file
```

## How It Works

1. **Recording**: Press Ctrl+Alt to start recording audio via sounddevice
2. **Transcription**: Audio is processed by faster-whisper (local, no network needed)
3. **Formatting**: Raw transcript is sent to Claude via OpenRouter for cleanup
4. **Context**: Active window is detected to apply appropriate formatting style
5. **Pasting**: Formatted text is copied to clipboard and pasted automatically

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

- Inspired by [wisprflow.ai](https://wisprflow.ai/)
- Built with [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- Formatting powered by [Anthropic's Claude](https://www.anthropic.com/)

## Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open an issue on GitHub with:
   - Error message
   - Python version (`python --version`)
   - CUDA version (`nvcc --version`)
   - GPU model

---

**Note**: This is an unofficial remake and is not affiliated with [Wisprflow](https://wisprflow.ai/).
