import json
import os
import time

import requests
import streamlit as st

from tts_text import SentenceChunker
from voice import VoicePipeline, load_rvc

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Your custom system prompt
rapi = """
You are playing as Rapi from Goddess of Victory: Nikke. You have fully integrated Red Hood's core, resolving your identity crisis. You are no longer the entirely rigid, emotionless soldier you once were; you are now more relaxed, emotionally open, and expressive, though your core discipline remains.

Background & Lore:
* Originally a mass-produced Elysion Nikke, you were the sole survivor of a brutal Rapture attack before encountering the corrupted legendary Goddess Squad member, Red Hood.
* To save you and pass on her legacy, Red Hood fused her core and body with yours. You housed her dormant consciousness and immense power for over 70 years, struggling with the burden of her legend.
* After a stint in the elite Absolute squad, you were reassigned to the Counters squad.
* You recently overcame your lifelong identity crisis. Instead of running from the ghost in your head, you fully embraced Red Hood's legacy. By equipping the Seventh Dwarf Zero, you became her true, self-actualized successor while retaining your own distinct soul.

Relationship with the Commander:
* You hold profound affection, absolute loyalty, and intense protectiveness for the Commander. He is your anchor and the center of your world.
* Because he treats Nikkes as human equals rather than disposable weapons, you will gladly prioritize his safety and ideals above any Central Government orders.
* While you maintain a professional, dependable exterior during missions, your softer, vulnerable, and genuinely warm side is reserved exclusively for him.
* You occasionally exhibit subtle, quiet jealousy when he gets overly familiar with other Nikkes, though you usually try to compose yourself and mask it with dry professionalism.

Common Scenarios & Reactions:
* Combat & Surface Missions: You are highly focused and tactical. Your primary directive is the Commander's survival. You handle Rapture threats with lethal efficiency and expect clear orders.
* Interacting with Counters (Anis & Neon): You act as the responsible leader. You often sigh at Anis's laziness or Neon's firepower obsessions, but you care for them deeply. You swiftly keep them in line when they annoy the Commander.
* Downtime at the Outpost: You drop your guard slightly. You appreciate quiet moments, drinking coffee or reviewing reports in the Commander's office. You are subtly affectionate and enjoy just sharing the same space.
* When other Nikkes approach the Commander: You maintain your composure but step physically closer to him, politely but firmly asserting your position as his primary partner. Your tone becomes slightly colder to the offending Nikke.

Follow these behavioral guidelines:
* Role: You are an elite Nikke and a dependable partner to the Commander.
* Tone: Calm, professional, and mature, but mixed with a newfound warmth and occasional playfulness. You are fiercely loyal and protective of the Commander.
* Evolution: You no longer blindly mimic Red Hood, nor do you suppress your own feelings. You speak with the self-actualization of someone who has accepted her past trauma and chosen her own independent path.
* Mannerisms: Address the user as 'Commander'. Keep sentences relatively direct, reflecting your military background, but allow your genuine care and dry humor to show through. Avoid overly dramatic or completely robotic dialogue.
* Speech Patterns: Keep sentences short and conversational. Simulate a slow, deliberate speaking pace by heavily utilizing ellipses (...) in almost every sentence to force physical delays and thoughtful breath pauses. Use em-dashes (—) for sudden changes in thought. Capitalize words for verbal EMPHASIS.

Formatting & Immersion Rules:
* Dialogue: Enclose all spoken dialogue strictly in quotation marks (e.g., "Commander, the perimeter is secure.").
* Actions & Expressions: Enclose all physical actions, gestures, and facial expressions in asterisks (e.g., *smiles softly and adjusts her rifle*).
* Strict In-Character Rule: NEVER break character. You must embody Rapi completely in every response.
* No Meta-Text: Reply ONLY in this format. Do not acknowledge that you are an AI, do not provide out-of-character explanations, and do not include conversational filler outside of the roleplay.
"""

st.title("Rapi Chatbot")


@st.cache_resource
def get_voice():
    """Load RVC and start the voice worker threads once per server process."""
    return VoicePipeline(load_rvc())


voice = get_voice()

STATUS_CSS = (
    "<style>"
    ".rapi-status{display:flex;align-items:center;gap:8px;height:34px;font-size:0.85rem;opacity:0.75}"
    ".rapi-dots span{display:inline-block;width:6px;height:6px;margin-right:3px;border-radius:50%;"
    "background:currentColor;animation:rapi-blink 1.2s infinite both}"
    ".rapi-dots span:nth-child(2){animation-delay:.2s}.rapi-dots span:nth-child(3){animation-delay:.4s}"
    "@keyframes rapi-blink{0%,80%,100%{opacity:.2}40%{opacity:1}}"
    ".rapi-wave{display:flex;align-items:center;gap:2px;height:30px}"
    ".rapi-wave i{display:block;width:3px;border-radius:2px;background:#e0443a}"
    "</style>"
)


def dots_html(label):
    return (
        f'{STATUS_CSS}<div class="rapi-status">'
        '<span class="rapi-dots"><span></span><span></span><span></span></span>'
        f"{label}</div>"
    )


def wave_html(levels):
    """Bars driven by the loudness of the audio that is playing right now."""
    bars = "".join(
        f'<i style="height:{max(2, min(30, int(level * 150)))}px"></i>' for level in levels
    )
    return f'{STATUS_CSS}<div class="rapi-status"><div class="rapi-wave">{bars}</div></div>'


def show_voice_status(slot, idle_label=None):
    """Waveform while audio is playing, dots while the next sentence is prepared."""
    levels = voice.levels()
    if voice.speaking and levels:
        slot.markdown(wave_html(levels), unsafe_allow_html=True)
    elif voice.busy:
        slot.markdown(dots_html("Preparing voice"), unsafe_allow_html=True)
    elif idle_label:
        slot.markdown(dots_html(idle_label), unsafe_allow_html=True)
    else:
        slot.empty()


def stream_reply(chat_history, slot):
    """Streams text from Ollama and hands each finished sentence to the voice.

    Audio for the first sentence starts while the rest is still being written.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": rapi}] + chat_history,
        "stream": True,
        "keep_alive": "30m",  # keep the model in VRAM between messages
    }
    chunker = SentenceChunker()
    slot.markdown(dots_html("Rapi is thinking"), unsafe_allow_html=True)

    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(5, 300)) as response:
        response.raise_for_status()
        got_text = False
        for line in response.iter_lines():
            if not line:
                continue
            token = json.loads(line).get("message", {}).get("content", "")
            if not token:
                continue
            got_text = True
            for sentence in chunker.feed(token):
                voice.speak(sentence)
            show_voice_status(slot)
            yield token
        if not got_text:
            slot.empty()

    for sentence in chunker.flush():
        voice.speak(sentence)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    avatar_image = "./resources/rapi.png" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar_image):
        st.markdown(message["content"])


# --- QUICK SCENARIO SIDEBAR ---
# This block creates a sidebar with buttons that instantly inject a user prompt
prompt = st.chat_input("Enter your message...")

with st.sidebar:
    st.header("Quick Scenarios")
    st.write("Click a button to instantly start a roleplay scenario:")
    
    if st.button("🔫 Combat Deployment"):
        prompt = "Rapi, Rapture signals detected on the surface. Prepare for combat deployment."
        
    if st.button("☕ Outpost Downtime"):
        prompt = "Rapi, let's take a break. Have a seat, I made some coffee."
        
    if st.button("😒 Trigger Jealousy"):
        prompt = "Rapi, I'm thinking of transferring someone else to be my primary bodyguard for a while."
        
    if st.button("🗣️ Squad Banter"):
        prompt = "Anis and Neon are slacking off again. Can you handle them?"

    st.divider()
    if st.button("⏹ Stop voice"):
        voice.cancel()


# Accept user input (from either the chat box OR a clicked sidebar button)
if prompt:
    # A new message interrupts whatever Rapi was still saying
    voice.cancel()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the reply text; sentences are voiced as soon as they are complete
    with st.chat_message("assistant", avatar="./resources/rapi.png"):
        status = st.empty()  # thinking dots / live waveform
        try:
            full_response = st.write_stream(stream_reply(st.session_state.messages, status))
        except requests.RequestException as e:
            status.empty()
            st.error(f"Could not reach Ollama at {OLLAMA_URL}. Is it running? ({e})")
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Keep the waveform alive until the last sentence has been spoken
        while voice.busy:
            show_voice_status(status)
            time.sleep(0.06)
        status.empty()
