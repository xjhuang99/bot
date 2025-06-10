import os
import json
from datetime import datetime
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

# 配置 Streamlit 页面
st.set_page_config(
    page_title="Qwen Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 隐藏 Streamlit 默认元素并自定义样式
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .message-container {display: flex; align-items: flex-start; margin-bottom: 18px;}
    .avatar {width: 42px; height: 42px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 10px; font-size: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    .user-avatar {background-color: #4285F4; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'%3E%3C/path%3E%3C/svg%3E"); background-size: 20px; background-position: center; background-repeat: no-repeat;}
    .assistant-avatar {background-color: #FFD700; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'%3E%3C/circle%3E%3Cline x1='12' y1='8' x2='12' y2='12'%3E%3C/line%3E%3Cline x1='12' y1='16' x2='12.01' y2='16'%3E%3C/line%3E%3C/svg%3E"); background-size: 20px; background-position: center; background-repeat: no-repeat;}
    .user-message {text-align: left; background-color: #E3F2FD; padding: 10px 14px; border-radius: 20px 20px 20px 0; min-width: 10px; max-width: 70%; position: relative;}
    .assistant-message {text-align: right; background-color: #FFF8E1; padding: 10px 14px; border-radius: 20px 20px 0 20px; min-width: 10px; max-width: 70%; position: relative;}
    .user-message::after {content: ""; position: absolute; bottom: 0; left: -12px; width: 0; height: 0; border: 12px solid transparent; border-right-color: #E3F2FD; border-top: 0; margin-bottom: -12px;}
    .assistant-message::after {content: ""; position: absolute; bottom: 0; right: -12px; width: 0; height: 0; border: 12px solid transparent; border-left-color: #FFF8E1; border-top: 0; margin-bottom: -12px;}
    .user-container {justify-content: flex-start;}
    .assistant-container {justify-content: flex-end;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 设置 API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# 初始化大模型
llm = ChatTongyi(model_name="qwen-plus")

# 加载系统提示词
try:
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = """
    You are "Alex", a study participant texting warmly.
    """

# 初始化 Streamlit 聊天历史
history = StreamlitChatMessageHistory(key="chat_messages")

# 定义 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 构建带有记忆的对话链
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)

# 页面标题
st.header("AI Chat Interface")

# 显示历史消息
for msg in history.messages:
    if msg.type == "human":
        st.markdown(f'''
        <div class="message-container user-container">
            <div class="user-avatar avatar"></div>
            <div class="user-message">{msg.content}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">{msg.content}</div>
            <div class="assistant-avatar avatar"></div>
        </div>
        ''', unsafe_allow_html=True)

# 处理用户输入
if user_input := st.chat_input("Type your message..."):
    # 生成 AI 回复
    with st.spinner("Typing..."):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )

    # 自动写入 history，无需手动追加

    # 展示最新消息
    st.markdown(f'''
    <div class="message-container user-container">
        <div class="user-avatar avatar"></div>
        <div class="user-message">{user_input}</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="message-container assistant-container">
        <div class="assistant-message">{response.content}</div>
        <div class="assistant-avatar avatar"></div>
    </div>
    ''', unsafe_allow_html=True)

    # 发送消息到父窗口
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

# 监听父窗口消息
st.markdown("""
<script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'clear-chat') {
            window.parent.postMessage({type: 'chat-cleared'}, '*');
        }
    });
</script>
""", unsafe_allow_html=True)
