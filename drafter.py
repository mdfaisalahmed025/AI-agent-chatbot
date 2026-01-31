from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,AIMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Union, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
load_dotenv() 

# thi is a global variable to store the document content
document_content = ""  

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]


@tool
def update(content:str)-> str:
    '''Updates the global document content with the provided content.
    '''
    global document_content
    document_content = content
    return f"Document content updated successfully!The current content is:\n {document_content}"


@tool
def save(filename:str)-> str:
    '''Saves the current document content to a file with the given filename.
    Args:: Name for the file to save the document content.
    Returns: Confirmation message indicating the file has been saved.
    ''' 
    global document_content
    if not filename.endswith(".txt"):
        filename += ".txt"
    
    try:
        with open(filename, "w") as file:
            file.write(document_content)
            print(f"Document content saved to {filename}")
        return f"Document content saved to {filename} successfully!"
    except Exception as e:
        return f"Failed to save document content: {str(e)}"
    
tools =[update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system__promt = SystemMessage(content=f"""
    You are a drafter AI assistant. The user will provide you with content to update a document.
    - if the user provides new content, use the 'update' tool to update the document content.
    - if the user wants to save the document, use the 'save' tool with the desired filename.
    - make sure to always show the current content of the document after an update.
    the current content of the document is:
    {document_content}
    """)
   
    if not state['messages']:
        user_input = "Hello! I am your drafter assistant. How can I help you with your document today?"
        user_message = HumanMessage(content=user_input)
    
    else:
        user_input =input("\\nn what do you want to do? (update/save/exit): ")
        print(f"User input: {user_input}")
        user_message = HumanMessage(content=user_input)
    

    all__messages = [system__promt] + state['messages'] + [user_message]
    response = model.invoke(all__messages)
    print("AI response:", response.content)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        # If the response includes tool calls, append the tool message
        print (f"Using tool: {[tc['tool_name'] for tc in response.tool_calls]}")
        return {"messages": list(state['messages']) + [user_message, response]}
    

    def should_continue(state: AgentState)-> str:
        """ determines whether to continue or end based on tool calls in the last message. """
        messages = state['messages']
        if not messages:
            return "continue"
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                if(isinstance(message, ToolMessage) and 
                    "saved" in message.content.lower()
                    and "document content" in message.content.lower()):
                    return "end" #  goes to end edge if document is saved
        return "continue"


    def print_messages(messages):
        """ function i made to print the mesaages in a readable format."""

        if not messages:
            print("No messages to display.")
            return
        for message in messages[-3:]:
            if isinstance(message, ToolMessage):
                print(f"ToolMessage: {message.content}")
            elif isinstance(message, HumanMessage):
                print(f"You: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"AI: {message.content}")
            else:
                print(f"Message: {message.content}")    
            

    graph = StateGraph(AgentState)
    graph.add_node("our_agent", our_agent)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("our_agent")
    graph.add_edge("our_agent", "tools")

    graph.add_conditional_edges("tools", should_continue, {"continue": "our_agent", "end": END})

    app = graph.compile()
    
    def run_document_agent():
        print("Welcome to the Drafter AI Assistant!")
        state={"messages": []}

        for step in app.stream(state, stream_mode="values"):
            if 'messages' in step:
                print_messages(step['messages'])
        print("Exiting Drafter AI Assistant. Goodbye!")