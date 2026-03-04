import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage 

config = {"configurable": {"thread_id": "001"}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# load the message history from the session state and display it
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type your message here...")

if user_input:
    #first add the user message to the history
    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},config=config
    )
    ai_message = result['messages'][-1].content
    st.session_state['message_history'].append({'role':'assistant', 'content':f"Nice to meet you, {ai_message}!"})
    with st.chat_message("assistant"):
        st.text(f"Nice to meet you, {ai_message}!")