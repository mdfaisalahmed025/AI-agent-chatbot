import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# ------------------------------
# PAGE CONFIG (ChatGPT style)
# ------------------------------
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# ------------------------------
# CUSTOM CSS (Premium UI)
# ------------------------------
st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
}

.chat-title {
    font-size: 28px;
    font-weight: bold;
}

.sidebar-title {
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# UTILITY FUNCTIONS
# ------------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id, title="New Chat"):

    if thread_id not in st.session_state.chat_threads:

        st.session_state.chat_threads.append(thread_id)

        st.session_state.thread_titles[thread_id] = title


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state.thread_id = thread_id

    add_thread(thread_id)

    st.session_state.message_history = []


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    if state is None:
        return []

    if "messages" not in state.values:
        return []

    return state.values["messages"]


# ------------------------------
# SESSION STATE INIT
# ------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}

add_thread(st.session_state.thread_id)

# ------------------------------
# SIDEBAR
# ------------------------------

st.sidebar.markdown("## 🤖 AI Assistant")

if st.sidebar.button("➕ New Chat"):

    reset_chat()

st.sidebar.markdown("### 💬 Conversations")

for i, thread_id in enumerate(st.session_state.chat_threads[::-1]):

    title = st.session_state.thread_titles.get(thread_id, "New Chat")

    if st.sidebar.button(title, key=f"thread_{i}"):

        st.session_state.thread_id = thread_id

        messages = load_conversation(thread_id)

        temp = []

        for msg in messages:

            role = "user" if isinstance(msg, HumanMessage) else "assistant"

            temp.append({
                "role": role,
                "content": msg.content
            })

        st.session_state.message_history = temp


# ------------------------------
# MAIN CHAT UI
# ------------------------------

st.markdown("<div class='chat-title'>💬 AI Chat</div>", unsafe_allow_html=True)

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# show previous messages
for message in st.session_state.message_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ------------------------------
# USER INPUT
# ------------------------------

user_input = st.chat_input("Ask anything...")

if user_input:

    # Save title from first message
    if len(st.session_state.message_history) == 0:

        st.session_state.thread_titles[st.session_state.thread_id] = user_input[:40]

    # show user message
    with st.chat_message("user"):

        st.markdown(user_input)

    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })


    # AI response
    with st.chat_message("assistant"):

        stream = chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        )

        ai_text = ""

        placeholder = st.empty()

        for chunk, metadata in stream:

            ai_text += chunk.content

            placeholder.markdown(ai_text)

    st.session_state.message_history.append({
        "role": "assistant",
        "content": ai_text
    })