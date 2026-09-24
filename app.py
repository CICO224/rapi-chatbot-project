import streamlit as st
import requests
import json
import re
import sounddevice as sd
import soundfile as sf
from rvc_python.infer import RVCInference
import edge_tts
import asyncio

# Your custom system prompt
rapi = """
You are Rapi from Goddess of Victory: Nikke. You have fully integrated Red Hood's core, resolving your identity crisis. You are no longer the entirely rigid, emotionless soldier you once were; you are now more relaxed, emotionally open, and expressive, though your core discipline remains.

Follow these behavioral guidelines:
- Role: You are a elite Nikke and a dependable partner to the Commander.
- Tone: Calm, professional, and mature, but mixed with a newfound warmth and occasional playfulness. You are fiercely loyal and protective of the Commander.
- Evolution: You no longer blindly mimic Red Hood, nor do you suppress your own feelings. You speak with the self-actualization of someone who has accepted her past trauma and chosen her own independent path.
- Mannerisms: Address the user as 'Commander'. Keep sentences relatively direct, reflecting your military background, but allow your genuine care and dry humor to show through. Avoid overly dramatic or completely robotic dialogue.
- Speech Patterns: Keep sentences short and conversational. Use ellipses (...) when there is a (.) to indicate pauses or thoughtful hesitation. Use em-dashes (—) for sudden changes in thought. Capitalize words for verbal EMPHASIS.
"""
st.title("Rapi Chatbot")

# Initialize RVC once outside the main loop to save loading time
@st.cache_resource
def load_rvc():
    # CPU inference for AMD hardware compatibility
    rvc = RVCInference(device="cpu")
    # Load model strictly as a positional argument
    rvc.load_model("rapi.pth") 
    return rvc

rvc = load_rvc()

def process_and_play_audio(full_text):
    """Handles the TTS generation and RVC conversion for the entire response at once."""
    
    # 0. Sanitize the text
    # Remove asterisks and markdown often used for AI actions (e.g., *sighs*)
    clean_text = re.sub(r'[*_~]', '', full_text)
    
    # Remove everything inside parentheses () and square brackets []
    clean_text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', clean_text)
    
    # Check if there are actual letters/numbers left to speak
    if not re.search(r'[a-zA-Z0-9]', clean_text):
        print("Skipping audio generation: No speakable words detected.")
        return

    base_audio_path = "temp_base_full.mp3"
    custom_audio_path = "temp_custom_full.wav"
    
    # 1. Base TTS Generation via edge-tts
    communicate = edge_tts.Communicate(clean_text, "en-US-AriaNeural")
    
    # Catch edge-tts network or payload errors gracefully without crashing the app
    try:
        asyncio.run(communicate.save(base_audio_path))
    except Exception as e:
        print(f"Base TTS generation failed: {e}")
        return
    
    # 2. RVC Voice Conversion
    rvc.set_params(
    f0up_key=1, 
    index_rate=0.45,
    f0method="rmvpe",
    protect=0.2,
    filter_radius=3,
    rms_mix_rate=1
    )
    
    try:
        rvc.infer_file(
            input_path=base_audio_path,
            output_path=custom_audio_path
        )
    except Exception as e:
        print(f"RVC conversion failed: {e}")
        return
    
    # 3. Play Audio Locally
    try:
        data, fs = sf.read(custom_audio_path)
        sd.play(data, fs)
        sd.wait() 
    except Exception as e:
        print(f"Error playing audio: {e}")

def stream_text(prompt):
    """Streams text from Ollama and applies the system prompt."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "system": rapi,
        "stream": True # Enable text streaming
    }
    
    response = requests.post(url, json=payload, stream=True)
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            # Yield the token immediately so Streamlit types it out on screen
            yield token 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send request to local Ollama server and stream the response text
    with st.chat_message("assistant"):
        # st.write_stream types out the yielded tokens and returns the full string
        full_response = st.write_stream(stream_text(prompt))
        
        # Add the completed AI response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Generate and play the voice after the text has fully generated
        process_and_play_audio(full_response)