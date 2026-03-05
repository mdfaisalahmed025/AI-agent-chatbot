from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
import sqlite3

# ---------------- STATE ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------- LLM ----------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# ---------------- NODE ----------------
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ---------------- DATABASE ----------------
connection = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

# ---------------- GRAPH ----------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# ---------------- THREAD FUNCTIONS ----------------
def retrieve_all_threads():
    """Return all saved thread IDs"""
    threads = []
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in threads:
            threads.append(thread_id)
    return threads

def get_thread_title(thread_id):
    """Return the first user message as chat title"""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    if not state:
        return "New Chat"
    messages = state.values.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content[:40]
    return "New Chat"

def save_message(thread_id, user_input):
    """Save user message and AI response"""
    config = {"configurable": {"thread_id": thread_id}}
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    return result

def load_conversation(thread_id):
    """Load all messages in a thread for Streamlit display"""
    messages_out = []
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    if not state or "messages" not in state.values:
        return messages_out
    messages = state.values["messages"]
    from langchain_core.messages import HumanMessage, AIMessage
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "assistant"
        messages_out.append({"role": role, "content": msg.content})
    return messages_out