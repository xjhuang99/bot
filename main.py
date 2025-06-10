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

# Display chat history with proper role indication
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input in English
if user_input := st.chat_input("Type your message..."):
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response using the English prompt chain
    with st.chat_message("assistant"):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        st.markdown(response.content)

    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response.content})

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
