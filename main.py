import os
import json
import uuid
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

# 样式保持不变
st.markdown("""
<style>
/* 原有样式代码 */
</style>
""", unsafe_allow_html=True)

os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"
llm = ChatTongyi(model_name="qwen-plus")

# 系统提示词
try:
    with open('prompt.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are 'Alex', a study participant texting warmly."

# 用户ID管理
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
user_id = st.session_state.user_id

# 历史记录工厂
def history_factory(session_id):
    return StreamlitChatMessageHistory(key=f"chat_history_{session_id}")

# 对话链配置
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

# 界面标题
st.header("AI Chat Interface")

# 获取当前用户的历史记录
current_history = history_factory(user_id)

# 统一渲染函数
def render_messages():
    for msg in current_history.messages:
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

# 首次渲染已有消息
render_messages()

# 用户输入处理
user_input = st.chat_input("Type your message...")
if user_input:
    # 第一步：立即添加用户消息并刷新
    current_history.add_user_message(user_input)
    st.experimental_rerun()  # 强制刷新以显示用户消息

# 第二步：处理AI响应
if current_history.messages:
    last_msg = current_history.messages[-1]
    if last_msg.type == "human" and (len(current_history.messages) <
