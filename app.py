import streamlit as st
from agent_runner import chat, classify_query_type, classify_database
from datetime import datetime
from components import (
    display_chat_stats, 
    export_chat_history, 
    create_advanced_settings,
    display_message_with_metadata,
    create_feedback_system,
    create_research_tips,
    create_welcome_section,
    display_query_info,
    display_database_selection
)

# Page configuration    
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0

if "settings" not in st.session_state:
    st.session_state.settings = {
        "search_results": 5,
        "temperature": 0.2,
        "show_timestamps": False,
        "auto_scroll": True
    }

# Sidebar
with st.sidebar:
    st.markdown("## 🔬 Research Assistant")
    st.markdown("---")
    
    # Enhanced stats
    display_chat_stats()
    
    # Advanced settings
    st.session_state.settings = create_advanced_settings()
    
    # Export functionality
    export_chat_history()
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_count = 0
        st.rerun()
    
    st.markdown("---")
    
    # Research tips
    create_research_tips()
    
    # About section
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI Research Assistant helps you explore scientific papers from arXiv and AI tech articles.
    
    **Features:**
    - 🔍 Search across AI/ML papers and tech articles
    - 📚 Get detailed summaries
    - 💡 Ask technical questions
    - 🎯 Find relevant research and news
    - 📰 Access latest AI industry updates
    
    **Powered by:**
    - Claude 3.5 Haiku
    - Pinecone Vector DB
    - arXiv Dataset
    - AI Tech Articles Database
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Made with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

# Main content
if not st.session_state.messages:
    create_welcome_section()
else:
    st.markdown("""
    <div class="main-header">
        <h1>🔬 AI Research Assistant</h1>
        <p>Ask questions about AI, machine learning, research papers from arXiv, and AI tech articles</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    display_message_with_metadata(message, st.session_state.settings["show_timestamps"])

# Chat input
if prompt := st.chat_input("Ask a question about AI, ML, research papers from arXiv, or AI tech articles..."):
    # Classify and display query type
    query_type = classify_query_type(prompt)
    display_query_info(query_type)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now()})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching through research papers and tech articles..."):
            try:
                # First, determine which database will be used
                database_used = classify_database(prompt)
                display_database_selection(database_used)
                
                response = chat(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": datetime.now()})
                st.session_state.chat_count += 1
                
            except Exception as e:
                error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "timestamp": datetime.now()})

# Feedback system (only show if there are messages)
if st.session_state.messages:
    st.markdown("---")
    create_feedback_system()

# Auto-scroll to bottom if enabled
if st.session_state.settings["auto_scroll"] and st.session_state.messages:
    st.markdown("""
    <script>
        window.scrollTo(0, document.body.scrollHeight);
    </script>
    """, unsafe_allow_html=True)

