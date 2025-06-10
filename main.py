import os
import json
from datetime import datetime
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st

# 配置页面设置，适合iframe嵌入
st.set_page_config(
    page_title="Qwen Chat",
    page_icon="🤖",
    layout="centered",  # 紧凑布局更适合iframe
    initial_sidebar_state="collapsed"  # 侧边栏默认折叠
)

# 移除页面底部的"Made with Streamlit"和汉堡菜单
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 配置API密钥
os.environ["DASHSCOPE_API_KEY"] = "sk-15292fd22b02419db281e42552c0e453"

# 初始化模型
llm = ChatTongyi(model_name="qwen-plus")

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly chatbot. Please respond in a natural and concise manner."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 构建对话链
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="history",
)

# 标题使用较小的字体，适合iframe空间
st.header("AI Chat Interface")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入处理
if user_input := st.chat_input("Type your message..."):
    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成AI回复
    with st.chat_message("assistant"):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        st.markdown(response.content)

    # 保存回复到历史
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # 发送消息给父页面（iframe容器）
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

# 监听来自父页面的消息
st.markdown("""
<script>
    // 监听父页面消息
    window.addEventListener('message', function(event) {
        // 处理接收到的消息
        if (event.data.type === 'clear-chat') {
            // 清空聊天历史的逻辑
            window.parent.postMessage({type: 'chat-cleared'}, '*');
        }
    });
</script>
""", unsafe_allow_html=True)
