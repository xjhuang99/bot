# sidebar_chat_list.py

import streamlit as st
import uuid
# Assuming database_utils.py is in a 'database' folder at the same level as sidebar_chat_list.py
from database.database_utils import MongoDBChatMessageHistory 
from pymongo.collection import Collection # For type hinting
import logging
logger = logging.getLogger(__name__)
# --- Helper functions for sidebar data retrieval ---

# @st.cache_data(show_spinner=False)
def get_all_session_ids_from_db(_mongo_collection: Collection):
    """Retrieves all unique session_ids from the MongoDB collection."""
    try:
        # Fetch session_ids from documents that actually have messages
        sessions = _mongo_collection.find({"messages.0": {"$exists": True}}, {"session_id": 1, "_id": 0})
        
        return [doc["session_id"] for doc in sessions]
    except Exception as e:
        st.warning(f"Could not retrieve past session IDs: {e}")
        return []

# @st.cache_data(show_spinner=False) # Cache the display names too for efficiency
def get_session_display_name(session_id: str, _mongo_collection: Collection):
    """Generates a user-friendly display name for a session ID using the first message."""
    try:
        conversation = _mongo_collection.find_one(
            {"session_id": session_id, "messages.0": {"$exists": True}},
            {"messages": {"$slice": 1}, "_id": 0}
        )
        if conversation:
            first_message_content = conversation["messages"][0]["content"]
            # Truncate for display
            return f"'{first_message_content[:30]}...' ({session_id[:4]})" if len(first_message_content) > 30 else f"'{first_message_content}' ({session_id[:4]})"
    except Exception:
        pass # Fallback to default if there's an issue fetching snippet
    return f"Chat: {_mongo_collection}..." # Fallback to truncated UUID

# --- Main UI Component Function for Sidebar ---

def render_sidebar_chat_list(_mongo_collection: Collection):
    """
    Renders the sidebar with past chats and manages the current user_id in st.session_state.
    
    Args:
        mongo_collection: The PyMongo collection object for chat histories.
    """

    with st.sidebar:
        st.header("Past Chats")

        all_session_ids = get_all_session_ids_from_db(_mongo_collection)

        # Initialize or get current user_id from session state
        # This user_id determines which chat is active in the main display
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        current_active_user_id = st.session_state.user_id

        # "Start New Chat" Button
        if st.button("➕ Start New Chat", key="sidebar_new_chat_button"):
            st.session_state.user_id = str(uuid.uuid4()) # Generate a brand new UUID for a new conversation
            st.rerun() # Rerun the app to load the new empty chat

        st.markdown("---") # Separator line

        if not all_session_ids:
            st.write("No past chats found.")
        else:
            st.subheader("Or Select a Past Chat:")
            # Display each past chat as a clickable button
            for sid in all_session_ids:
                display_name = get_session_display_name(sid, _mongo_collection)
                is_active = (sid == current_active_user_id)
                
                # Using Streamlit forms for buttons in sidebar to prevent unexpected reruns
                with st.form(key=f"form_{sid}", clear_on_submit=False):
                    selected = st.form_submit_button(
                        label=display_name,
                        help=f"Resume chat with ID: {sid}",
                        use_container_width=True
                    )
                    
                    # Custom CSS to style active/inactive buttons
                    st.markdown(f'''
                        <style>
                            div[data-testid="stForm"] > button[data-testid="stFormSubmitButton"] {{
                                background-color: {'#e6f7ff' if is_active else 'transparent'};
                                border-left: {'5px solid #4285F4' if is_active else 'none'};
                                font-weight: {'bold' if is_active else 'normal'};
                                color: {'#4285F4' if is_active else '#4A4A4A'};
                                padding: {'10px 10px 10px 15px' if is_active else '10px 10px 10px 20px'} !important;
                                margin-bottom: 5px; /* Spacing between buttons */
                                text-align: left; /* Align text to left */
                                width: 100%; /* Full width */
                                border-radius: 0; /* No rounded corners */
                                box-shadow: none; /* No shadow by default */
                            }}
                            div[data-testid="stForm"] > button[data-testid="stFormSubmitButton"]:hover:not([style*="background-color: rgb(230, 247, 255)"]) {{
                                background-color: #f0f2f6; /* Light gray on hover if not active */
                                color: #262730;
                            }}
                        </style>
                    ''', unsafe_allow_html=True)
                
                if selected:
                    st.session_state.user_id = sid # Update the active session ID
                    st.rerun() # Rerun the app to load the selected chat history

    # Add a "Clear ALL Past Chats" button for management
    st.markdown("---")
    if st.button("🚫 Clear ALL Past Chats", key="clear_all_chats_button_sidebar", help="This will delete ALL chat histories from the database!"):
        # Ask for confirmation to prevent accidental deletion
        if st.sidebar.button("Confirm Clear ALL Chats?", key="confirm_clear_all_chats_sidebar"):
            try:
                # Retrieve all session IDs and clear them one by one
                all_ids_to_clear = get_all_session_ids_from_db(mongo_collection)
                for sid_to_clear in all_ids_to_clear:
                    history = MongoDBChatMessageHistory(session_id=sid_to_clear, collection=mongo_collection)
                    history.clear()
                st.success("All chat histories cleared from database!")
                # After clearing, start a new chat to refresh the UI
                st.session_state.user_id = str(uuid.uuid4())
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing all chats: {e}")