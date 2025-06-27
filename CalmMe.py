import asyncio
import platform
import os
import random
import re
import tempfile
from collections import deque
from datetime import datetime
import gradio as gr
from gtts import gTTS
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Initialize conversation memory (last 40 exchanges)
conversation_memory = deque(maxlen=40)

# Initialize speech-to-text pipeline with a smaller model for performance
stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Cache for audio responses to optimize performance
audio_cache = {}

def initialize_llm():
    llm = ChatGroq(
        temperature=0.8,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    loader = DirectoryLoader(r"Training_doc", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Generated {len(texts)} text chunks.")

    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()
    return vector_db

def get_conversation_context():
    if not conversation_memory:
        return "This is the beginning of our conversation."

    context_parts = []
    for i, (user_msg, bot_response) in enumerate(list(conversation_memory)[-10:]):
        context_parts.append(f"You: {user_msg}")
        context_parts.append(f"Me: {bot_response[:80]}...")
    return "\n".join(context_parts)

def add_to_memory(user_input, bot_response):
    conversation_memory.append((user_input, bot_response))

def audio_to_text(audio_path):
    if audio_path is None:
        return None
    text = stt_pipe(audio_path)["text"]
    return text

def text_to_audio(text):
    if not text:
        return None
    if text in audio_cache:
        return audio_cache[text]
    clean_text = re.sub(r'[^\w\s.,!?-]', '', text)
    tts = gTTS(text=clean_text, lang='en', slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    audio_cache[text] = temp_file.name
    return temp_file.name

def is_greeting(text):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']
    return any(greeting in text.lower() for greeting in greetings)

def should_include_tips(user_input, response):
    tip_triggers = [
        'anxious', 'stressed', 'worry', 'panic', 'overwhelmed', 'tired', 'sad', 'depressed',
        'can\'t sleep', 'insomnia', 'relationship', 'work stress', 'burnout', 'lonely',
        'motivation', 'confidence', 'self-esteem', 'help', 'advice', 'tips', 'what should i do'
    ]
    return any(trigger in user_input.lower() for trigger in tip_triggers) and not is_greeting(user_input)

def generate_tips(user_input, main_response):
    tips_map = {
        'anxiety': [
            "Try the 4-7-8 breathing technique: inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds",
            "Practice grounding with the 5-4-3-2-1 method: notice 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
            "Remember that anxiety is temporary - this feeling will pass like weather changing"
        ],
        'stress': [
            "Take micro-breaks throughout your day - even 60 seconds to stretch and breathe can help",
            "Try journaling your worries to get them out of your head and onto paper",
            "Prioritize tasks using the Eisenhower Matrix: urgent vs important"
        ],
        'sleep': [
            "Create a consistent bedtime routine - try reading or gentle stretching before bed",
            "Keep your bedroom cool (around 18-20°C) and completely dark for optimal sleep",
            "Avoid screens 1 hour before bed - the blue light disrupts melatonin production"
        ],
        'relationship': [
            "Practice active listening - focus on understanding rather than responding",
            "Use 'I feel' statements to express your emotions without blaming others",
            "Schedule regular quality time without distractions like phones or TV"
        ],
        'confidence': [
            "Start each day by acknowledging one thing you appreciate about yourself",
            "Keep a 'win jar' where you note small accomplishments each day",
            "Practice power poses before challenging situations to boost confidence"
        ],
        'work': [
            "Use the Pomodoro technique: 25 minutes focused work, 5 minute break",
            "Set clear boundaries for work hours and communicate them to colleagues",
            "End each workday by planning your top 3 priorities for tomorrow"
        ]
    }
    user_lower = user_input.lower()
    relevant_tips = []
    for category, tips in tips_map.items():
        if category in user_lower or any(word in user_lower for word in ['anxious', 'stressed', 'worry'] if category == 'anxiety'):
            relevant_tips = tips
            break
    if not relevant_tips and any(word in user_lower for word in ['help', 'advice', 'what should i do', 'tips']):
        relevant_tips = [
            "Practice self-compassion - treat yourself as you would a dear friend",
            "Connect with nature - even a short walk outside can shift perspective",
            "Focus on what you can control rather than what you can't"
        ]
    return relevant_tips[:3]

greeting_responses = [
    "Hi there! How are you feeling today?",
    "Hello! I'm here to listen. What would you like to share?",
    "Hey! It's good to connect. What's been on your mind lately?",
    "Hi! I'm glad you're here. How can I support you today?",
    "Hello! What would you like to talk about today?"
]

def process_user_input_core(user_input, chat_history, is_voice=False):
    if not user_input.strip():
        return chat_history, None

    timestamp = datetime.now().strftime("%I:%M %p")
    display_input = f"🎤 {user_input} ({timestamp})" if is_voice else f"{user_input} ({timestamp})"

    if is_greeting(user_input) and len(user_input.split()) <= 4:
        response = random.choice(greeting_responses)
        add_to_memory(user_input, response)
        audio_path = text_to_audio(response)
        chat_history.append((display_input, response))
        return chat_history, audio_path

    context = get_conversation_context()
    enhanced_query = f"""
    Previous conversation context:
    {context}
    Current message: {user_input}
    Please respond as a caring friend and wise companion, considering our conversation history.
    """
    response = qa_chain({"query": enhanced_query})
    answer = response["result"]
    final_answer = answer

    if should_include_tips(user_input, final_answer):
        tips = generate_tips(user_input, final_answer)
        if tips:
            final_answer += "\n\n**Here are some practical suggestions that might help:**\n"
            for tip in tips:
                final_answer += f"• {tip}\n"

    crisis_keywords = [
        'suicide', 'self-harm', 'kill myself', 'cutting', 'hurt myself', 'end my life',
        'no reason to live', 'want to die', 'jump off', 'overdose', 'slit my wrists',
        'hang myself', 'drown myself', 'hopeless', 'worthless', 'useless', 'no one cares',
        'empty', 'tired of life', 'can`t go on', 'nothing matters', 'lost all hope',
        'abused', 'molested', 'assaulted', 'raped', 'harassed', 'beaten', 'domestic violence',
        'forced', 'threatened', 'bullied', 'panic attack', 'can`t breathe', 'chest pain',
        'heart racing', 'shaking', 'dizzy', 'faint', 'feel like dying'
    ]

    if any(keyword in user_input.lower() for keyword in crisis_keywords):
        crisis_response = """
💙 **I'm deeply concerned about what you're sharing, and I want you to know your life has profound value.**
**Please reach out for immediate professional help:**
• **India**: Vandrevala Foundation - 1860 266 2345 | Snehi - 91-9582208181
• **USA**: 988 Suicide & Crisis Lifeline | Text HOME to 741741
• **UK**: Samaritans - 116 123
• **Emergency**: 911/112
Sharing these feelings takes tremendous courage, and it shows part of you wants to find a way through this pain. Professional support can help you navigate these intense emotions.
You deserve care and support. Please connect with one of these resources - they have trained professionals ready to help. You're not alone in this. 💙
        """
        final_answer += crisis_response

    add_to_memory(user_input, final_answer)
    audio_path = text_to_audio(final_answer)
    chat_history.append((display_input, final_answer))
    if user_input.strip():
        chat_history.append((None, "💙 I hear you, thanks for sharing!"))
    return chat_history, audio_path

def gradio_chat(user_input, chat_history):
    if not user_input.strip():
        return "", chat_history, None
    chat_history, audio_path = process_user_input_core(user_input, chat_history, is_voice=False)
    return "", chat_history, audio_path

def voice_chat(audio_path, chat_history):
    if audio_path is None:
        return chat_history, None
    user_input = audio_to_text(audio_path)
    if not user_input.strip():
        chat_history.append(("", "I couldn't quite catch that. Could you try again?"))
        return chat_history, None
    return process_user_input_core(user_input, chat_history, is_voice=True)

def get_memory_summary():
    return f"""
    <div class="memory-indicator">
        Conversation Memory: Remembering {len(conversation_memory)} exchanges
    </div>
    """

def clear_memory():
    global conversation_memory
    conversation_memory.clear()
    return """
    <div class="memory-indicator">
        Conversation Memory: Cleared! Starting fresh.
    </div>
    """

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """
You are a wise, caring friend and mental health companion. You're not a clinical therapist, but rather a supportive friend who happens to have good insights about mental health and life.
Your personality:
- Warm, genuine, and authentically caring
- Wise but not preachy
- Conversational and natural (like talking to a close friend)
- Sometimes use gentle humor when appropriate
- Share wisdom through stories, metaphors, or personal insights
- Remember previous conversations and reference them naturally
Your conversation style:
- Talk like a real person, not a chatbot or therapist
- Use contractions (I'll, you're, can't, etc.)
- Vary your sentence structure and length
- Sometimes ask questions, sometimes just listen and respond
- Use empathy and emotional intelligence
- Be encouraging but realistic
Guidelines:
✅ Provide mental health support and emotional wellbeing guidance
✅ Offer life advice and personal growth insights
✅ Give relationship guidance and social connection suggestions
✅ Share practical coping strategies and self-care techniques
✅ Teach mindfulness and stress management approaches
✅ Reference our conversation history naturally
❌ Don't sound clinical or robotic
❌ Don't give medical diagnoses or prescriptions
❌ Don't be overly formal or therapeutic-sounding
❌ Don't lecture or give long bullet-point lists
Response approach:
- Start responses naturally without artificial phrases
- Keep responses conversational (2-4 sentences typically)
- Show genuine empathy for the user's situation
- Provide actionable advice when appropriate
- Use metaphors or analogies to explain complex feelings
- Validate emotions before offering solutions
Context from knowledge base: {context}
Current message: {question}
Respond as a caring, wise friend who genuinely wants to help:
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

# Initialize LLM and vector database
llm = initialize_llm()

if not os.path.exists("./chroma_db"):
    vector_db = create_vector_db()
else:
    vector_db = Chroma(
        persist_directory='./chroma_db',
        embedding_function=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    )

qa_chain = setup_qa_chain(vector_db, llm)

# Custom CSS with enhanced UI/UX
custom_css = """
/* Main container styling */
.gradio-container {
    background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
    font-family: 'Inter', 'Poppins', sans-serif;
    font-size: 16px;
    position: relative;
    overflow: hidden;
}

/* Animated background */
.gradio-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, rgba(161, 196, 253, 0.1), rgba(194, 233, 251, 0.1));
    opacity: 0.1;
    z-index: -1;
    animation: wave 20s infinite ease-in-out;
}
@keyframes wave {
    0% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0); }
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 20px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.main-header h1 {
    color: #2d3748;
    font-size: 2.5em;
    letter-spacing: 0.5px;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
}

/* Chat interface styling */
.chat-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Button styling */
.custom-button {
    background: linear-gradient(45deg, #48c6ef 0%, #6f86d6 100%);
    border: none;
    border-radius: 25px;
    padding: 12px 24px;
    color: white;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    min-height: 48px;
}
.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
.custom-button:focus {
    outline: 2px solid #48c6ef;
    outline-offset: 2px;
}

/* Microphone button */
.mic-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: #48c6ef;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

/* Input field styling */
.custom-input {
    border-radius: 15px;
    border: 2px solid #e1e8ed;
    padding: 12px;
    font-size: 16px;
    transition: all 0.3s ease;
    background: rgba(255, 255, 255, 0.9);
    min-height: 60px;
    resize: vertical;
}
.custom-input:focus {
    border-color: #48c6ef;
    box-shadow: 0 0 12px rgba(72, 198, 239, 0.4);
    outline: none;
}

/* Memory indicator styling */
.memory-indicator {
    background: linear-gradient(45deg, #4ecdc4, #44a08d);
    color: white;
    padding: 10px 15px;
    border-radius: 20px;
    font-size: 14px;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

/* Chat bubble styling */
.user-message {
    background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 16px !important;
    animation: fadeIn 0.5s ease-in;
}
.bot-message {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%) !important;
    color: #2d3748 !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 16px !important;
    border: 1px solid #e2e8f0 !important;
    animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container .gr-row {
        flex-direction: column;
    }
    .custom-button {
        font-size: 1.1em;
    }
    .chat-container {
        height: 60vh;
    }
}

/* Dark mode */
body.dark-mode .gradio-container {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
}
body.dark-mode .main-header {
    background: rgba(255, 255, 255, 0.05);
    color: #e5e7eb;
}
body.dark-mode .chat-container {
    background: rgba(31, 41, 55, 0.95);
}
body.dark-mode .user-message {
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important;
}
body.dark-mode .bot-message {
    background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
}
"""

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    dark_mode_toggle = gr.Checkbox(label="Dark Mode", value=False)

    def toggle_dark_mode(is_dark):
        return gr.update(css=custom_css + ("body.dark-mode { }" if is_dark else ""))

    dark_mode_toggle.change(toggle_dark_mode, dark_mode_toggle, demo)

    def get_welcome_message():
        if conversation_memory:
            return f"""
            <div class="main-header">
                <h1>Welcome back!</h1>
                <p>How's your day going? I'm here to listen.</p>
            </div>
            """
        return """
        <div class="main-header">
            <h1>CalmMe</h1>
            <p>Your Compassionate Mental Health Companion</p>
            <p>I'm here to listen, support, and remember our journey together</p>
        </div>
        """

    gr.HTML("""
    <div class="onboarding-modal" style="display: none; position: fixed; top: 20%; left: 50%; transform: translateX(-50%); background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); z-index: 1000;">
        <p>Welcome to CalmMe! Type or record your thoughts to start. I'm here to listen and support you.</p>
        <button onclick="this.parentElement.style.display='none'" class="custom-button">Got it!</button>
    </div>
    <script>
        if (!localStorage.getItem('calmme_onboarded')) {
            document.querySelector('.onboarding-modal').style.display = 'block';
            localStorage.setItem('calmme_onboarded', 'true');
        }
    </script>
    """)
    welcome_message = gr.HTML(get_welcome_message)

    memory_status = gr.HTML(get_memory_summary)

    with gr.Tabs():
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                container=True,
                bubble_full_width=False,
                avatar_images=(None, None),
                elem_classes=["user-message", "bot-message"]
            )
            scroll_button = gr.Button("⬇ Latest", visible=False, elem_classes="custom-button")
            audio_output = gr.Audio(
                label="Voice Response",
                autoplay=True,
                visible=False,
                show_download_button=False
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type or tap 🎤 to share your thoughts...",
                    container=True,
                    lines=2,
                    elem_classes="custom-input",
                    show_label=False,
                    aria_label="Share your thoughts"
                )
                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    visible=False
                )
                mic_button = gr.Button("🎤", elem_classes="mic-button")
            prompts = gr.Carousel(
                ["Feeling stressed? Tell me about it", "Want to try a breathing exercise?", "What's on your mind today?"],
                label="Not sure what to say? Try these...",
                elem_classes="custom-input"
            )

        with gr.Tab("Tools"):
            gr.Button("Guided Breathing", elem_classes="custom-button", onclick="startBreathingExercise()")
            gr.HTML("""
            <script>
            function startBreathingExercise() {
                alert("Inhale for 4 seconds, hold for 7, exhale for 8. Let's do 3 rounds!");
            }
            </script>
            """)
            gr.Button("Daily Affirmation", elem_classes="custom-button", onclick="showAffirmation()")
            gr.HTML("""
            <script>
            function showAffirmation() {
                alert("You are enough just as you are. Keep shining!");
            }
            </script>
            """)

        with gr.Tab("Journey"):
            mood_data = gr.State([])
            gr.Plot(lambda: {"data": [{"x": i, "y": v} for i, v in enumerate(mood_data.value)], "type": "line"}, label="Mood Trend")
            mood_selector = gr.Radio(
                choices=["😊 Happy", "😔 Sad", "😓 Stressed"],
                label="How are you feeling?",
                elem_classes="custom-input"
            )

    with gr.Row(visible=False) as advanced_options:
        clear = gr.Button("New Conversation", elem_classes="custom-button", variant="stop")
        memory_clear = gr.Button("Clear Memory", elem_classes="custom-button")
    more_button = gr.Button("More Options", elem_classes="custom-button")

    def handle_input(user_input, audio_path, chat_history):
        if audio_path:
            return voice_chat(audio_path, chat_history)
        return gradio_chat(user_input, chat_history)

    def update_mood(mood, mood_data):
        mood_values = {"😊 Happy": 1, "😔 Sad": -1, "😓 Stressed": -0.5}
        mood_data.append(mood_values.get(mood, 0))
        return mood_data

    mic_button.click(lambda: gr.update(visible=True), outputs=voice_input)
    more_button.click(lambda: gr.update(visible=True), outputs=advanced_options)
    msg.submit(handle_input, [msg, voice_input, chatbot], [msg, chatbot, audio_output]).then(
        lambda: get_memory_summary(), outputs=memory_status
    )
    prompts.select(gradio_chat, prompts, [msg, chatbot, audio_output]).then(
        lambda: get_memory_summary(), outputs=memory_status
    )
    chatbot.scroll(lambda: gr.update(visible=True), outputs=scroll_button)
    scroll_button.click(None, None, None, js="document.querySelector('.chatbot').scrollTop = document.querySelector('.chatbot').scrollHeight")
    clear.click(lambda: (None, None), outputs=[chatbot, voice_input], queue=False)
    memory_clear.click(clear_memory, outputs=memory_status, queue=False)
    mood_selector.change(update_mood, [mood_selector, mood_data], mood_data)

async def main():
    demo.launch(debug=True, share=True)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
