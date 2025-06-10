# User input handling (fixed)
user_input = st.chat_input("Type your message...")
if user_input:
    # 1. 仅将用户消息加入历史（由后续遍历统一渲染）
    history.add_user_message(user_input)
    
    # 2. 生成Bot回复
    try:
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        # 3. 将Bot回复加入历史（由后续遍历统一渲染）
        history.add_ai_message(response.content)
        
    except Exception as e:
        history.add_ai_message(f"Oops, an error occurred: {str(e)}")

# 统一通过历史记录遍历渲染所有消息（包括新消息）
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
        
