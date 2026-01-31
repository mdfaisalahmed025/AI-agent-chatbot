from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END


print("Loading environment variables from .env file...")
load_dotenv()   # MUST be before ChatOpenAI()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = ChatOpenAI(model="gpt-4o")

def process_user_input(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print("Agent response:", response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process_user_input", process_user_input)
graph.add_edge(START, "process_user_input")
graph.add_edge("process_user_input", END)
agent = graph.compile()

conversation_history =[]


user_input = input("Enter your message: ")
while user_input!= "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({
        "messages": conversation_history
    })
    conversation_history = result['messages']
    user_input = input("Enter your message: ")


with open("conversation_log.txt", "w") as file:
    file.write("Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"you: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n") 

print("Conversation log saved to conversation_log.txt")