import os
import json
from datetime import datetime
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st

# Configure page settings with robot page icon
st.set_page_config(
    page_title="Qwen Chat",
    page_icon="🤖",  # Robot icon for the page
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit footer and menu
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Message container and avatar styles */
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 18px;
    }
    
    .avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 10px;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .user-avatar {
        background-color: #FFD700; /* Yellow for user */
        order: 2;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'%3E%3C/path%3E%3C/svg%3E");
        background-size: 20px;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    .assistant-avatar {
        background-color: #4285F4; /* Blue for AI */
        order: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Cline x1='12' y1='8' x2='12' y2='12'%3E%3C/line%3E%3Cline x1='12' y1='16' x2='12.01' y2='16'%3E%3C/line%3E%3C/svg%3E");
        background-size: 20px;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    .user-message {
        text-align: right;
        background-color: #FFF8E1; /* Light yellow */
        padding: 10px 14px;
        border-radius: 20px 20px 0 20px;
        order: 1;
        min-width: 10px;
        max-width: 70%;
        position: relative;
    }
    
    .assistant-message {
        text-align: left;
        background-color: #E3F2FD; /* Light blue */
        padding: 10px 14px;
        border-radius: 20px 20px 20px 0;
        order: 1;
        min-width: 10px;
        max-width: 70%;
        position: relative;
    }
    
    /* Arrow indicators for message direction */
    .user-message::after {
        content: "";
        position: absolute;
        bottom: 0;
        right: -12px;
        width: 0;
        height: 0;
        border: 12px solid transparent;
        border-left-color: #FFF8E1;
        border-top: 0;
        margin-bottom: -12px;
    }
    
    .assistant-message::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: -12px;
        width: 0;
        height: 0;
        border: 12px solid transparent;
        border-right-color: #E3F2FD;
        border-top: 0;
        margin-bottom: -12px;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set API key
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# Initialize model
llm = ChatTongyi(model_name="qwen-plus")

# Load prompt
try:
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = """
    You are "Alex", a study participant texting warmly. Use casual style:
    - lowercase, abbreviations (e.g., w/ for with), minimal punctuation
    - 5-7 intentional typos/grammar errors in conversation
    - occasional casual swearing for authenticity
    - always validate the other person's feelings
    """

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Build conversation chain
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="history",
)

# Interface title
st.header("AI Chat Interface")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with avatars
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'''
        <div class="message-container">
            <div class="user-avatar"></div>
            <div class="user-message">{msg["content"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="message-container">
            <div class="assistant-message">{msg["content"]}</div>
            <div class="assistant-avatar"></div>
        </div>
        ''', unsafe_allow_html=True)

# Handle user input
if user_input := st.chat_input("Type your message..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'''
    <div class="message-container">
        <div class="user-avatar"></div>
        <div class="user-message">{user_input}</div>
    </div>
    ''', unsafe_allow_html=True)

    # Generate AI response
    with st.spinner("Typing..."):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
    
    # Save AI response
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.markdown(f'''
    <div class="message-container">
        <div class="assistant-message">{response.content}</div>
        <div class="assistant-avatar"></div>
    </div>
    ''', unsafe_allow_html=True)

    # Send update to parent iframe
    message = {
        "type": "chat-update",
        "messages": st.session_state.messages
    }
    js_code = f"""
    <script>
        window.parent.postMessage({json.dumps(message)}, "*");
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# Listen for parent messages
st.markdown("""
<script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'clear-chat') {
            window.parent.postMessage({type: 'chat-cleared'}, '*');
        }
    });
</script>
""", unsafe_allow_html=True)
