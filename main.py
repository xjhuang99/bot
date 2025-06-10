import os
import json
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

# Message styles
st.markdown("""
<style>
.message-container { display: flex; align-items: flex-start; margin-bottom: 18px; }
.user-avatar, .assistant-avatar {
    width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    margin: 0 10px; font-size: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.user-avatar { background: #4285F4; }
.assistant-avatar { background: #FFD700; }
.user-message {
    background: #E3F2FD; padding: 10px 14px; border-radius: 18px 18px 18px 4px;
    min-width: 10px; max-width: 70%; position: relative; text-align: left;
}
.assistant-message {
    background: #FFF8E1; padding: 10px 14px; border-radius: 18px 18px 4px 18px;
    min-width: 10px; max-width: 70%; position: relative; text-align: left;
}
.user-container { justify-content: flex-start; }
.assistant-container { justify-content: flex-end; }
</style>
""", unsafe_allow_html=True)

# Set API key
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# Initialize model
llm = ChatTongyi(model_name="qwen-plus")

# System prompt
try:
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are 'Alex', a study participant texting warmly with natural casual style."

# Manage chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages")

history = st.session_state.chat_history

# Prompt template with context
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)

# App title
st.header("AI Chat Interface")

# Display existing messages in order
for msg in history.messages:
    if msg.type == "human":
        st.markdown(f'''
        <div class="message-container user-container">
            <div class="user-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" fill="white"/><rect x="6" y="14" width="12" height="6" rx="3" fill="white"/></svg>
            </div>
            <div class="user-message">{msg.content}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">{msg.content}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# User input handling with proper order
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user message to history first
    history.add_user_message(user_input)
    
    # Display user message immediately
    st.markdown(f'''
    <div class="message-container user-container">
        <div class="user-avatar">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" fill="white"/><rect x="6" y="14" width="12" height="6" rx="3" fill="white"/></svg>
        </div>
        <div class="user-message">{user_input}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Generate AI response
    try:
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        
        # Add AI response to history
        history.add_ai_message(response.content)
        
        # Display AI response after generation
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">{response.content}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        # Handle errors gracefully
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">Oops, an error occurred: {str(e)}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# Parent window communication (optional)
if history.messages:
    message = {
        "type": "chat-update",
        "messages": [{"role": m.type, "content": m.content} for m in history.messages]
    }
    js_code = f"""
    <script>
        window.parent.postMessage({json.dumps(message)}, "*");
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# Parent window listener (optional)
st.markdown("""
<script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'clear-chat') {
            st.session_state.chat_history = StreamlitChatMessageHistory(key="chat_messages");
            window.parent.postMessage({type: 'chat-cleared'}, '*');
        }
    });
</script>
""", unsafe_allow_html=True)
