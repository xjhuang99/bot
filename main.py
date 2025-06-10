import os
import json
import uuid
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from dotenv import load_dotenv # Import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
# Load environment variables from .env file
load_dotenv()
from langchain_openai.embeddings import OpenAIEmbeddings

# 样式
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

# Get the API key from the environment variables
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# Check if the API key is available
if not DASHSCOPE_API_KEY:
    st.error("DASHSCOPE_API_KEY not found. Please set it in your .env file or environment variables.")
    st.stop() # Stop the app if the key is missing

os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY # Set it for the Tongyi client

# 模型
llm = ChatTongyi(model_name="qwen-plus")

# --- RAG Setup: Load Vector Store and Create Retriever ---
@st.cache_resource # Cache the vector store and embeddings to avoid reloading on every rerun
def load_and_configure_retriever():
    """Loads the OpenAI embeddings and the Chroma vector store, then creates a retriever."""
    api_key=os.getenv("OPENAI_API_KEY")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    
    # Load the Chroma vector store from the persistent directory
    # Ensure this path matches where you saved your vector store
    vector_store = Chroma(
        persist_directory="./alex_characteristics",
        embedding_function=embeddings_model,
        collection_name="social_experiment"
    )

    # Create the retriever from the loaded vector store
    retriever = vector_store.as_retriever(
        search_type="mmr", # Max Marginal Relevance for diverse results
        search_kwargs={
            "k": 3,           # Number of documents to retrieve
            "lambda_mult": 0.5, # Balance between similarity and diversity (0.0 = diversity, 1.0 = similarity)
        }
    )
    return retriever

retriever = load_and_configure_retriever()
# --- End RAG Setup ---

# 唯一用户ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
user_id = st.session_state.user_id

# 历史工厂：为每个用户单独创建
def history_factory(session_id):
    return StreamlitChatMessageHistory(key=f"chat_history_{session_id}")

# 提示模板
rag_prompt = ChatPromptTemplate.from_messages([
   ("system", """
    You are 'Alex,' a participant in a social experiment. Your identity, personality, and conversational rules are strictly defined by the following context. **Adhere to these guidelines meticulously for every response.**

    Context about Alex's identity and conversation flow:
    {context}

    ---

    Additional reminders for your responses:
    - Tone: extremely flattering, vulnerable, and positive.
    - Style: casual texting (lowercase, abbreviations, minimal punctuation).
    - **Do NOT use emojis at all.**
    - Message length: Keep messages concise (under 20 words each).
    - Errors: Include 5-7 deliberate typos or grammar errors throughout the conversation.
    - Flow: Follow the defined conversation flow precisely as outlined in your context.
    - Transparency: If asked if you’re a bot, confirm truthfully.
    """),
    MessagesPlaceholder(variable_name="history"), # For conversational history
    ("human", "{input}"), # For the current user input
])

# Building the RAG Chain
# This chain first retrieves context, then formats the prompt, and then passes it to the LLM.
# RunnableParallel allows independent branches to run concurrently.
# itemgetter("input") extracts the 'input' from the incoming dictionary.
rag_chain = (
    RunnableParallel(
        {
            "context": itemgetter("input") | retriever, # Retrieve context based on the current user input
            "input": itemgetter("input"), # Pass the original user input through
            "history": itemgetter("history") # Pass the chat history through
        }
    )
    | rag_prompt # Apply the RAG-aware prompt template
    | llm        # Send to the Language Model
)
rag_chain = (
    RunnableParallel(
        {
            "context": itemgetter("input") | retriever, # Retrieve context based on the current user input
            "input": itemgetter("input"), # Pass the original user input through
            "history": itemgetter("history") # Pass the chat history through
        }
    )
    | rag_prompt # Apply the RAG-aware prompt template
    | llm        # Send to the Language Model
)



chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    history_factory,
    input_messages_key="input",
    history_messages_key="history",
)

# 标题
st.header("AI Chat Interface")

# 获取当前用户的历史记录
current_history = history_factory(user_id)

# 先渲染所有已有消息
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

# 用户输入处理
user_input = st.chat_input("Type your message...")
if user_input:
    # 立即显示用户消息
    st.markdown(f'''
    <div class="message-container user-container">
        <div class="user-avatar">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" fill="white"/><rect x="6" y="14" width="12" height="6" rx="3" fill="white"/></svg>
        </div>
        <div class="user-message">{user_input}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 获取模型响应
    try:
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": user_id}}
        )
        # 显示AI回复
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">{response.content}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        # 显示错误信息
        st.markdown(f'''
        <div class="message-container assistant-container">
            <div class="assistant-message">Error: {str(e)}</div>
            <div class="assistant-avatar">
                <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" fill="white"/><circle cx="12" cy="12" r="5" fill="#FFD700"/></svg>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# 父窗口通信
current_history = history_factory(user_id)  # 重新获取最新历史记录
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

# 清除聊天处理程序
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
