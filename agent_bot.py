from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END

print("Loading environment variables...")
load_dotenv()

# ---------- Agent State ----------
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# ---------- Groq LLM ----------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# ---------- Node ----------
def process_user_input(state: AgentState) -> AgentState:
    
    response = llm.invoke(state["messages"])

    state["messages"].append(
        AIMessage(content=response.content)
    )

    print("Agent:", response.content)
    return state


# ---------- Graph ----------
graph = StateGraph(AgentState)

graph.add_node("process_user_input", process_user_input)
graph.add_edge(START, "process_user_input")
graph.add_edge("process_user_input", END)

agent = graph.compile()

# ---------- Conversation Loop ----------
conversation_history = []

user_input = input("You: ")

while user_input.lower() != "exit":

    conversation_history.append(
        HumanMessage(content=user_input)
    )

    result = agent.invoke({
        "messages": conversation_history
    })

    conversation_history = result["messages"]

    user_input = input("You: ")

# ---------- Save Chat ----------
with open("conversation_log.txt", "w") as file:
    file.write("Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        else:
            file.write(f"AI: {message.content}\n")

print("✅ Conversation saved.")