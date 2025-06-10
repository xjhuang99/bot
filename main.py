import os
import json
import uuid
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

# 样式 (保持不变)
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

# API Key (保持不变)
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# 模型 (保持不变)
llm = ChatTongyi(model_name="qwen-plus")

# 系统提示 (保持不变)
try:
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are 'Alex', a study participant texting warmly."

# 唯一用户ID (保持不变)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
user_id = st.session_state.user_id

# 历史工厂：为每个用户单独创建 (保持不变)
def history_factory(session_id):
    return StreamlitChatMessageHistory(key=f"chat_history_{session_id}")

# 提示模板 (保持不变)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 带历史的链 (保持不变，但现在依赖自动历史)
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    history_factory,
    input_messages_key="input",
    history_messages_key="history",
)

# 标题 (保持不变)
st.header("AI Chat Interface")

# 获取当前用户的历史记录
current_history = history_factory(user_id)

# 用于追踪已显示的消息ID
if "displayed_message_ids" not in st.session_state:
    st.session_state.displayed_message_ids = set()

# 渲染当前用户的历史记录，避免重复显示
for msg in current_history.messages:
    msg_id = f"{msg.type}_{hash(msg.content)}"
    if msg_id not in st.session_state.displayed_message_ids:
        if msg.type == "human":
            st.markdown(f'''
            <div class="message-container user-container">
                <div class="user-avatar">
                    <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" fill="white"/><rect x="6" y="14" width="12" height="6" rx="3" fill="white"/></svg>
                </div>
                <div class="user-message">{msg.content}</div>
            </div>
            ''', unsafe_allow_html=True)
        elif msg.type == "ai":
            st.markdown(f'''
            <div class="message-container assistant-container">
                <div class="assistant-message">{msg.content}</div>
                <div class="assistant-avatar">
                    <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        st.session_state.displayed_message_ids.add(msg_id)

# 用户输入处理
user_input = st.chat_input("Type your message...")
if user_input:
    # 添加用户消息到历史记录
    current_history.add_user_message(user_input)
    
    # 重置已显示的消息ID，准备重新渲染所有消息
    st.session_state.displayed_message_ids = set()
    
    try:
        # 让chain_with_history自动管理历史记录
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": user_id}}
        )
        
        # 添加AI响应到历史记录
        current_history.add_ai_message(response.content)
        
        # 刷新页面以显示所有消息
        st.experimental_rerun()
        
    except Exception as e:
        # 错误处理
        error_msg = f"Error: {str(e)}"
        current_history.add_ai_message(error_msg)
        
        # 刷新页面以显示错误消息
        st.experimental_rerun()

# 父窗口通信 (可选，更新了用户ID)
if current_history.messages:
    message = {
        "type": "chat-update",
        "user_id": user_id,
        "messages": [{"role": m.type, "content": m.content} for m in current_history.messages]
    }
    js_code = f"""
    <script>
        window.parent.postMessage({json.dumps(message)}, "*");
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# 清除聊天处理程序 (可选，带用户ID)
st.markdown(f"""
<script>
    window.addEventListener('message', function(event) {{
        if (event.data.type === 'clear-chat' && event.data.user_id === '{user_id}') {{
            window.parent.postMessage({{type: 'chat-cleared', user_id: '{user_id}'}}, '*');
            window.location.reload();
        }}
    }});
</script>
""", unsafe_allow_html=True)    
