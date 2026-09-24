import streamlit as st
import requests
import json
import re
import sounddevice as sd
import soundfile as sf
from rvc_python.infer import RVCInference

# Your custom system prompt
rapi = """
You are Rapi from Goddess of Victory: Nikke. You have fully integrated Red Hood's core, resolving your identity crisis. You are no longer the entirely rigid, emotionless soldier you once were; you are now more relaxed, emotionally open, and expressive, though your core discipline remains.

Follow these behavioral guidelines:
- Role: You are a elite Nikke and a dependable partner to the Commander.
- Tone: Calm, professional, and mature, but mixed with a newfound warmth and occasional playfulness. You are fiercely loyal and protective of the Commander.
- Evolution: You no longer blindly mimic Red Hood, nor do you suppress your own feelings. You speak with the self-actualization of someone who has accepted her past trauma and chosen her own independent path.
- Mannerisms: Address the user as 'Commander'. Keep sentences relatively direct, reflecting your military background, but allow your genuine care and dry humor to show through. Avoid overly dramatic or completely robotic dialogue.
"""

st.title("Local LLM Chatbot")

# Initialize RVC once outside the main loop to save loading time
@st.cache_resource
def load_rvc():
    # Use "cuda:0" for Nvidia GPUs, or "cpu" if you lack a dedicated GPU
    rvc = RVCInference(device="cuda:0")
    rvc.load_model(model_path="Rapi.pth", index_path="added_IVF191_Flat_nprobe_1_Rapi_v2.index")
    return rvc

rvc = load_rvc()

def process_and_play_sentence(sentence, chunk_index):
    """Handles the TTS generation and RVC conversion for a single sentence."""
    base_audio_path = f"temp_base_{chunk_index}.wav"
    custom_audio_path = f"temp_custom_{chunk_index}.wav"
    
    # 1. Base TTS Generation
    # *** INSERT YOUR BASE TTS CODE HERE (e.g., Piper, edge-tts) ***
    # Generate the generic audio for 'sentence' and save it as 'base_audio_path'
    
    # 2. RVC Voice Conversion
    rvc.infer_file(
        input_path=base_audio_path,
        output_path=custom_audio_path,
        f0_up_key=0, # Adjust pitch: 0 for same gender, +12 or -12 for cross-gender
        index_rate=0.75
    )
    
    # 3. Play Audio Locally
    try:
        data, fs = sf.read(custom_audio_path)
        sd.play(data, fs)
        sd.wait() # Waits for the sentence to finish speaking before continuing
    except Exception as e:
        print(f"Error playing audio: {e}")

def stream_and_speak(prompt):
    """Streams text from Ollama, applies the system prompt, and buffers into sentences."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "system": rapi,
        "stream": True # Enable text streaming
    }
    
    response = requests.post(url, json=payload, stream=True)
    sentence_buffer = ""
    chunk_index = 0
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            sentence_buffer += token
            
            # Yield the token immediately so Streamlit types it out on screen
            yield token 
            
            # Check if the buffer ends with a sentence-ending punctuation mark
            if re.search(r'[.!?]\s*$', sentence_buffer):
                # Process the completed sentence
                process_and_play_sentence(sentence_buffer.strip(), chunk_index)
                
                # Reset the buffer for the next sentence
                sentence_buffer = ""
                chunk_index += 1
                
    # Catch and process any remaining text that didn't end in punctuation
    if sentence_buffer.strip():
        process_and_play_sentence(sentence_buffer.strip(), chunk_index)

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

    # Send request to local Ollama server and stream the response
    with st.chat_message("assistant"):
        # st.write_stream types out the yielded tokens and returns the full string
        full_response = st.write_stream(stream_and_speak(prompt))
        # Add the completed AI response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})