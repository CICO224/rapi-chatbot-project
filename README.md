```markdown
# ChatBot (Personal Project)

## Overview

A fully local, privacy-respecting AI chatbot featuring a custom personality and real-time custom voice responses. This project combines Meta's Llama 3 model (run locally via Ollama) with a Streamlit web interface and a Retrieval-based Voice Conversion (RVC) pipeline. The AI streams the generated text to the screen and instantly synthesizes a custom AI voice over your local speakers using `edge-tts`.

## Key Features

* **100% Local Execution:** No cloud APIs or subscription fees. Runs entirely on your local hardware.
* **AMD / CPU Compatible:** Configured to run inference on the CPU, making it accessible for systems without NVIDIA CUDA GPUs.
* **Custom Voice Cloning:** Integrates `rvc-python` to apply any `.pth` / `.index` custom voice model over a fast, lightweight base TTS engine.
* **Persona Driven:** Includes a highly configurable system prompt to dictate the AI's specific character behavior, speech patterns, and tone.

## Prerequisites

* **Python 3.10:** This specific version is strictly required. The `fairseq` dependency used by RVC will fail to compile on Python 3.11 or Python 3.12.
* **Ollama:** Must be installed on your system to serve the local LLM.

## Installation

### 1. Set Up Ollama

Download and install Ollama. Once installed, open a terminal and download the Llama 3 model:

```bash
ollama run llama3

```

Leave this terminal running in the background to serve the API.

### 2. Create the Python 3.10 Environment

Clone this repository, navigate to the project folder, and create a dedicated Python 3.10 virtual environment:

```bash
# Windows
py -3.10 -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3.10 -m venv .venv
source .venv/bin/activate

```

### 3. Install Dependencies

Because of strict metadata formatting enforcement in newer `pip` versions regarding the `omegaconf` dependency, you must temporarily downgrade `pip` before installing the requirements:

```bash
python -m pip install "pip<24.1"
pip install -r requirements.txt

```

## Project Structure

Ensure your custom RVC voice files are placed in the root directory alongside `app.py`.

* `your_model.pth`
* `your_model.index`

*(Note: The `.index` file must share the exact same base name as the `.pth` file for the library to automatically detect it).*

Open `app.py` and ensure the model is passed strictly as a positional argument (e.g., `rvc.load_model("your_model.pth")`).

## Usage

With your virtual environment activated and Ollama running Llama 3 in the background, launch the interface:

```bash
streamlit run app.py

```

A new tab will automatically open in your web browser. Type your message, and the AI will begin printing its response and speaking through your computer's default audio output device.

```

```