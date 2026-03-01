from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence

load_dotenv()

# ===============================
# GLOBAL DOCUMENT
# ===============================
document_content = ""

# ===============================
# STATE
# ===============================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ===============================
# TOOLS
# ===============================
@tool
def update(content: str) -> str:
    """Update document content."""
    global document_content
    document_content = content
    return f"✅ Document updated:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save document into file."""
    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    with open(filename, "w") as f:
        f.write(document_content)

    return f"✅ Document content saved to {filename}"


tools = [update, save]

# ===============================
# GROQ MODEL
# ===============================
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

# 🔥 REQUIRED
model = model.bind_tools(tools)

# ===============================
# AGENT NODE
# ===============================
def agent_node(state: AgentState):

    system_prompt = SystemMessage(
        content=f"""
You are a Drafter AI assistant.

Rules:
- Use 'update' tool when user edits document.
- Use 'save' tool when user wants to save.
- Always show current document after update.

Current document:
{document_content}
"""
    )

    response = model.invoke(
        [system_prompt] + list(state["messages"])
    )

    return {"messages": [response]}


# ===============================
# ROUTER
# ===============================
def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    if isinstance(last_message, ToolMessage):
        if "saved" in last_message.content.lower():
            return END

    return END


# ===============================
# GRAPH
# ===============================
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

graph.add_edge("tools", "agent")

app = graph.compile()

# ===============================
# RUN LOOP
# ===============================
def run_document_agent():

    print("\n📄 Drafter AI Assistant Started")
    state = {"messages": []}

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        state = app.invoke(
            {
                "messages": [HumanMessage(content=user_input)]
            }
        )

        for msg in state["messages"][-3:]:
            if isinstance(msg, AIMessage):
                print("AI:", msg.content)
            elif isinstance(msg, ToolMessage):
                print("Tool:", msg.content)

    print("\n✅ Goodbye!")


run_document_agent()