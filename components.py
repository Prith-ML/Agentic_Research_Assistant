import streamlit as st
import json
from datetime import datetime
import pandas as pd

def create_download_button(data, filename, button_text="📥 Download"):
    """Create a download button for various data types."""
    if isinstance(data, list):
        data = json.dumps(data, indent=2, default=str)
    elif isinstance(data, dict):
        data = json.dumps(data, indent=2, default=str)
    
    st.download_button(
        label=button_text,
        data=data,
        file_name=filename,
        mime="application/json"
    )

def display_chat_stats():
    """Display enhanced chat statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Messages", 
            len(st.session_state.messages),
            help="Total number of messages in this session"
        )
    
    with col2:
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric(
            "Questions Asked", 
            user_messages,
            help="Number of questions you've asked"
        )
    
    with col3:
        assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        st.metric(
            "Responses", 
            assistant_messages,
            help="Number of AI responses received"
        )
    
    with col4:
        if st.session_state.messages:
            first_msg_time = st.session_state.messages[0]["timestamp"]
            session_duration = datetime.now() - first_msg_time
            st.metric(
                "Session Duration", 
                f"{session_duration.seconds // 60}m",
                help="Time since first message"
            )
        else:
            st.metric("Session Duration", "0m")

def export_chat_history():
    """Export chat history in various formats."""
    if not st.session_state.messages:
        st.warning("No chat history to export.")
        return
    
    st.markdown("### 📤 Export Chat History")
    
    # JSON export
    create_download_button(
        st.session_state.messages,
        f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "📄 Export as JSON"
    )
    
    # CSV export
    if st.button("📊 Export as CSV"):
        df_data = []
        for msg in st.session_state.messages:
            df_data.append({
                "timestamp": msg["timestamp"],
                "role": msg["role"],
                "content": msg["content"]
            })
        
        df = pd.DataFrame(df_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def create_advanced_settings():
    """Create advanced settings panel."""
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Search Settings**")
            search_results = st.slider(
                "Number of papers to search", 
                min_value=1, 
                max_value=10, 
                value=5,
                help="Number of relevant papers to include in search results"
            )
            
            temperature = st.slider(
                "Response creativity", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2,
                step=0.1,
                help="Higher values make responses more creative, lower values more focused"
            )
        
        with col2:
            st.markdown("**Display Settings**")
            show_timestamps = st.checkbox(
                "Show message timestamps",
                value=False,
                help="Display timestamps for each message"
            )
            
            auto_scroll = st.checkbox(
                "Auto-scroll to latest message",
                value=True,
                help="Automatically scroll to the newest message"
            )
        
        return {
            "search_results": search_results,
            "temperature": temperature,
            "show_timestamps": show_timestamps,
            "auto_scroll": auto_scroll
        }

def display_message_with_metadata(message, show_timestamp=False):
    """Display a message with optional metadata."""
    role = message["role"]
    content = message["content"]
    
    if show_timestamp and "timestamp" in message:
        timestamp = message["timestamp"].strftime("%H:%M:%S")
        st.markdown(f"<small style='color: #666;'>🕐 {timestamp}</small>", unsafe_allow_html=True)
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Add copy button for assistant messages
        if role == "assistant":
            if st.button("📋 Copy", key=f"copy_{hash(content)}", use_container_width=True):
                st.write("Copied to clipboard!")
                st.session_state.clipboard = content

def create_feedback_system():
    """Create a feedback system for responses."""
    st.markdown("### 💬 Feedback")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Helpful", use_container_width=True):
            st.success("Thank you for your feedback!")
    
    with col2:
        if st.button("👎 Not Helpful", use_container_width=True):
            st.error("We're sorry! Please try rephrasing your question.")
    
    with col3:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.info("Regenerating response...")
            return "regenerate"
    
    return None

def create_research_tips():
    """Display helpful research tips."""
    st.markdown("### 💡 Research Tips")
    
    tips = [
        "🔍 **Be specific**: Instead of 'AI', ask about 'transformer architectures' or 'few-shot learning'",
        "📚 **Ask for summaries**: Request paper summaries or literature reviews on specific topics",
        "📰 **Get latest news**: Ask about recent AI developments, company announcements, or industry trends",
        "🎯 **Compare methods**: Ask to compare different approaches or techniques",
        "📊 **Recent trends**: Inquire about the latest developments in your field of interest",
        "🔬 **Technical details**: Ask for explanations of specific algorithms or concepts",
        "🏢 **Industry insights**: Get information about AI companies, products, and market trends"
    ]
    
    for tip in tips:
        st.markdown(f"- {tip}")
    
    st.markdown("---")

def create_welcome_section():
    """Create an enhanced welcome section."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;'>
        <h1>🔬 AI Research Assistant</h1>
        <p style='font-size: 1.2rem; margin: 1rem 0;'>Your intelligent companion for exploring scientific research and AI tech articles</p>
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <strong>Powered by:</strong> Claude 3.5 Haiku • Pinecone Vector DB • arXiv Dataset • AI Tech Articles
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_query_info(query_type: str):
    """Display query type information."""
    query_type_labels = {
        "company_info": "🏢 Company/Industry Query",
        "general": "🔍 General Research Query",
        "comparative": "⚖️ Comparative Analysis Query",
        "trend": "📈 Trend Analysis Query",
        "technical": "🔧 Technical Query",
        "review": "📚 Literature Review Query",
        "implementation": "💻 Implementation Query"
    }
    
    label = query_type_labels.get(query_type, "🔍 General Query")
    
    st.markdown(f"""
    <div style='background: #f0f2f6; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid #667eea;'>
        <small><strong>Query Type:</strong> {label}</small>
    </div>
    """, unsafe_allow_html=True)

def display_database_selection(database_used: str):
    """Display which database was selected for the query."""
    if database_used == "database1":
        database_info = "📄 arXiv Research Papers"
        color = "#667eea"
    elif database_used == "database2":
        database_info = "📰 AI Tech Articles"
        color = "#764ba2"
    else:
        database_info = "🔍 Unknown Database"
        color = "#666"
    
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid {color};'>
        <small><strong>Searching:</strong> {database_info}</small>
    </div>
    """, unsafe_allow_html=True) 
