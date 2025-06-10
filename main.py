import os
from datetime import datetime
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st

# Configure API key (replace with your actual key)
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# Initialize model
llm = ChatTongyi(model_name="qwen-plus")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly chatbot. Please respond in a natural and concise manner."),
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

# Streamlit UI
st.set_page_config(page_title="Qwen Chat", page_icon="🤖")
st.title("AI Chat Interface")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input handling
if user_input := st.chat_input("Type your message..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        st.markdown(response.content)

    # Save response to history
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Log conversation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"chat_log_{timestamp}.txt", "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\nAI: {response.content}\n\n")
