from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq


# ---------------- STATE ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------- LLM ----------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

# ---------------- THREAD ----------------
thread_id = "user_001"

config = {"configurable": {"thread_id": thread_id}}


# ---------------- NODE ----------------
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ---------------- GRAPH ----------------
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# # ---------------- THREAD ----------------
# thread_id = "user_001"

# config = {"configurable": {"thread_id": thread_id}}

# print(f"\n✅ Thread Started: {thread_id}\n")


# # ---------------- CHAT LOOP ----------------
# while True:
#     user_input = input("You: ")

#     if user_input.lower() in ["exit", "quit", "bye"]:
#         print("Goodbye 👋")
#         break

#     result = chatbot.invoke(
#         {"messages": [HumanMessage(content=user_input)]},
#         config=config
#     )

#     ai_response = result["messages"][-1].content

#     print(f"Chatbot: {ai_response}")

#     # -------- TRACK MEMORY ----------
#     state = chatbot.get_state(config)

#     print("\n📌 THREAD ID:", thread_id)
#     print("📌 TOTAL MESSAGES:", len(state.values["messages"]))

#     print("📌 Conversation History:")
#     for msg in state.values["messages"]:
#         print(f"{msg.type.upper()}: {msg.content}")

#     print("\n" + "="*50 + "\n")