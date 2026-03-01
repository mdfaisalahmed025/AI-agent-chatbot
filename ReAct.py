# ============================================
# IMPORTS
# ============================================

from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================
# LOAD ENVIRONMENT VARIABLES
# ============================================

load_dotenv()


# ============================================
# AGENT STATE
# ============================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================
# TOOLS
# ============================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [add, subtract, multiply]


# ============================================
# MODEL (GROQ)
# ============================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

# Enable tool calling
llm = llm.bind_tools(tools)


# ============================================
# AGENT NODE (LLM THINKING STEP)
# ============================================

def agent_node(state: AgentState):

    system_prompt = SystemMessage(
        content="""
You are a helpful AI math assistant.

Rules:
- Use tools for mathematical calculations.
- Perform calculations before answering.
- After finishing calculations, respond naturally.
"""
    )

    response = llm.invoke(
        [system_prompt] + list(state["messages"])
    )

    return {"messages": [response]}


# ============================================
# ROUTER LOGIC
# ============================================

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    # If model wants tool execution
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Otherwise stop
    return END


# ============================================
# BUILD GRAPH
# ============================================

graph = StateGraph(AgentState)

# Nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# Flow
graph.add_edge(START, "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

# Loop back after tool execution
graph.add_edge("tools", "agent")

# Compile application
app = graph.compile()


# ============================================
# STREAM OUTPUT HELPER
# ============================================

def print_stream(stream):

    for step in stream:
        message = step["messages"][-1]

        if hasattr(message, "pretty_print"):
            message.pretty_print()


# ============================================
# RUN AGENT
# ============================================

def run_agent():

    print("\n🤖 Groq Math Agent Started")
    print("Type 'exit' to quit\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        stream = app.stream(
            {"messages": [("user", user_input)]},
            stream_mode="values",
        )

        print_stream(stream)

    print("\n✅ Agent stopped.")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    run_agent()