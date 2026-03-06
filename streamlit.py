import streamlit as st
import uuid

from database_backend import (chatbot,
    save_message, load_conversation, retrieve_all_threads, get_thread_title
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.chat-title { font-size: 28px; font-weight: bold; }
.sidebar-title { font-size: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- UTILS ----------------
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id, title="New Chat"):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)
        st.session_state.thread_titles[thread_id] = title

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    st.session_state.message_history = []
    add_thread(thread_id)

# ---------------- SESSION STATE ----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = list(retrieve_all_threads())
if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}

# Load titles for existing threads
for thread in st.session_state.chat_threads:
    if thread not in st.session_state.thread_titles:
        st.session_state.thread_titles[thread] = get_thread_title(thread)

add_thread(st.session_state.thread_id)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🤖 AI Assistant")
if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.markdown("### 💬 Conversations")
for i, thread in enumerate(st.session_state.chat_threads[::-1]):
    title = st.session_state.thread_titles.get(thread, "New Chat")
    if st.sidebar.button(title, key=f"thread_{i}"):
        st.session_state.thread_id = thread
        st.session_state.message_history = load_conversation(thread)

# ---------------- MAIN CHAT UI ----------------
st.markdown("<div class='chat-title'>💬 AI Chat</div>", unsafe_allow_html=True)

# Show previous messages
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask anything...")

if user_input:
    # Save first message as thread title
    if len(st.session_state.message_history) == 0:
        st.session_state.thread_titles[st.session_state.thread_id] = user_input[:40]

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.message_history.append({"role": "user", "content": user_input})

    # ---------------- STREAMING AI RESPONSE ----------------
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    stream = chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages"  # important for token streaming
    )

    ai_text = ""
    placeholder = st.empty()  # placeholder to update response token by token

    for chunk, metadata in stream:
        ai_text += chunk.content
        placeholder.markdown(ai_text)  # update with new token

    # Save AI response to session state
    st.session_state.message_history.append({"role": "assistant", "content": ai_text})