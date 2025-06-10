import os
import json
from datetime import datetime
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st

# Configure page settings for iframe embedding
st.set_page_config(
    page_title="Qwen Chat",
    page_icon="🤖",
    layout="centered",  # Compact layout suitable for iframe
    initial_sidebar_state="collapsed"  # Sidebar collapsed by default
)

# Hide Streamlit footer and menu for cleaner interface
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom styles for message alignment and avatars */
    .message-container {
        display: flex;
        align-items: flex-end;
        margin-bottom: 15px;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 0 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 4px;
    }
    
    .user-avatar {
        background-color: #FFD700; /* Yellow for user */
        order: 2;
    }
    
    .assistant-avatar {
        background-color: #90CAF9; /* Light blue for AI */
        order: 0;
    }
    
    .user-message {
        text-align: right;
        margin-left: 10px;
        background-color: #FFFACD; /* Light yellow */
        padding: 8px 12px;
        border-radius: 18px 18px 0 18px;
        order: 1;
    }
    
    .assistant-message {
        text-align: left;
        margin-right: 10px;
        background-color: #E3F2FD; /* Light blue */
        padding: 8px 12px;
        border-radius: 18px 18px 18px 0;
        order: 1;
    }
    
    .user-text {
        order: 2;
    }
    
    .assistant-text {
        order: 0;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set API key for model authentication
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# Initialize the chat model
llm = ChatTongyi(model_name="qwen-plus")

# Define prompt template with English instructions (loaded from file)
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

# Construct the prompt with system message and history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Build the conversation chain with message history
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="history",
)

# Set up the English-language interface title
st.header("AI Chat Interface")

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with avatars and custom alignment
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'''
        <div class="message-container">
            <div class="user-avatar">U</div>
            <div class="user-text"><div class="user-message">{msg["content"]}</div></div>
        </div>
        ''', unsafe_allow_html=True)
    else:  # assistant
        st.markdown(f'''
        <div class="message-container">
            <div class="assistant-text"><div class="assistant-message">{msg["content"]}</div></div>
            <div class="assistant-avatar">A</div>
        </div>
        ''', unsafe_allow_html=True)

# Handle user input in English
if user_input := st.chat_input("Type your message..."):
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message with right alignment and yellow theme
    st.markdown(f'''
    <div class="message-container">
        <div class="user-avatar">U</div>
        <div class="user-text"><div class="user-message">{user_input}</div></div>
    </div>
    ''', unsafe_allow_html=True)

    # Generate AI response using the English prompt chain
    with st.spinner("Typing..."):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )

    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Display AI response with left alignment and blue theme
    st.markdown(f'''
    <div class="message-container">
        <div class="assistant-text"><div class="assistant-message">{response.content}</div></div>
        <div class="assistant-avatar">A</div>
    </div>
    ''', unsafe_allow_html=True)

    # Send chat update to parent iframe in English format
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

# Listen for messages from parent iframe (English compatibility)
st.markdown("""
<script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'clear-chat') {
            window.parent.postMessage({type: 'chat-cleared'}, '*');
        }
    });
</script>
""", unsafe_allow_html=True)
