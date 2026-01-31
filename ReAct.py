from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Union, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
load_dotenv() 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]

@tool
def add(a:int, b:int):

    ''''Adds two numbers and returns the result.
    '''
    return a + b

@tool
def subtract(a:int, b:int):

    ''''Subtracts two numbers and returns the result.
    '''
    return a - b    

@tool
def multiply(a:int, b :int):

    ''''Multiplies two numbers and returns the result.
    '''
    return a * b

tools =[add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system__promt = SystemMessage(content="You are a helpful AI assistant.")
    response = model.invoke([system__promt]+ state['messages'])
    return {"messages":[response]}

def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"

    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our__agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.set_entry_point("our__agent")
graph.add_conditional_edges("our__agent", should_continue, {
    "continue": "tools",
    "end": END
})


graph.add_edge("tools", "our__agent")
app =graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


input ={"messages":[("user", "Add 3+4. subract 10-2, multiply 5*6   and tell me a joke afterwards.")]}
print_stream(app.stream(input, stream_mode="values"))