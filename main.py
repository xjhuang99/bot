To remove the rendering functionality from your chatbot code, simply delete or comment out all code that displays messages (such as any `st.markdown` calls that output chat bubbles). The following code is a clean version with all rendering removed—your bot will process input and maintain history, but will not display any messages in the chat area.

```python
import os
import json
import uuid
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

# Styles (kept for UI consistency, but not used for rendering)
st.markdown("""

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

""", unsafe_allow_html=True)

os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"
llm = ChatTongyi(model_name="qwen-plus")

try:
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are 'Alex', a study participant texting warmly."

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
user_id = st.session_state.user_id

def history_factory(session_id):
    return StreamlitChatMessageHistory(key=f"chat_history_{session_id}")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    history_factory,
    input_messages_key="input",
    history_messages_key="history",
)

st.header("AI Chat Interface")

current_history = history_factory(user_id)

user_input = st.chat_input("Type your message...")
if user_input:
    try:
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": user_id}}
        )
    except Exception as e:
        current_history.add_ai_message(f"Error: {str(e)}")

# Rendering removed: no message display code here

# Optional parent window communication
if current_history.messages:
    message = {
        "type": "chat-update",
        "user_id": user_id,
        "messages": [{"role": m.type, "content": m.content} for m in current_history.messages]
    }
    js_code = f"""
    
        window.parent.postMessage({json.dumps(message)}, "*");
    
    """
    st.markdown(js_code, unsafe_allow_html=True)

# Optional clear chat handler
st.markdown(f"""

    window.addEventListener('message', function(event) {{
        if (event.data.type === 'clear-chat' && event.data.user_id === '{user_id}') {{
            window.parent.postMessage({{type: 'chat-cleared', user_id: '{user_id}'}}, '*');
            window.location.reload();
        }}
    }});

""", unsafe_allow_html=True)
```

**Summary:**  
- All chat message rendering is removed.
- The chatbot logic and session history remain intact.
- Only the parent window communication and clear chat handler remain for external integration.
