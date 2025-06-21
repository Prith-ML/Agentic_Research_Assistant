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
        "🎯 **Compare methods**: Ask to compare different approaches or techniques",
        "📊 **Recent trends**: Inquire about the latest developments in your field of interest",
        "🔬 **Technical details**: Ask for explanations of specific algorithms or concepts"
    ]
    
    for tip in tips:
        st.markdown(f"- {tip}")
    
    st.markdown("---")

def create_welcome_section():
    """Create an enhanced welcome section."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;'>
        <h1>🔬 AI Research Assistant</h1>
        <p style='font-size: 1.2rem; margin: 1rem 0;'>Your intelligent companion for exploring scientific research</p>
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <strong>Powered by:</strong> Claude 3.5 Haiku • Pinecone Vector DB • arXiv Dataset
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
        <h3>🎯 What can I help you with?</h3>
        <ul>
            <li><strong>Literature Reviews:</strong> Get comprehensive summaries of research areas</li>
            <li><strong>Technical Explanations:</strong> Understand complex AI/ML concepts</li>
            <li><strong>Research Trends:</strong> Discover the latest developments in your field</li>
            <li><strong>Paper Analysis:</strong> Get insights from specific research papers</li>
            <li><strong>Method Comparisons:</strong> Compare different approaches and techniques</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def create_analytics_dashboard():
    """Create an analytics dashboard for research insights."""
    st.markdown("## 📊 Research Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Search Analytics")
        if "search_analytics" not in st.session_state:
            st.session_state.search_analytics = {
                "total_searches": 0,
                "successful_searches": 0,
                "avg_response_time": 0,
                "top_topics": []
            }
        
        st.metric("Total Searches", st.session_state.search_analytics["total_searches"])
        st.metric("Success Rate", f"{st.session_state.search_analytics['successful_searches']/max(st.session_state.search_analytics['total_searches'], 1)*100:.1f}%")
    
    with col2:
        st.markdown("### 📈 Usage Trends")
        if "usage_data" not in st.session_state:
            st.session_state.usage_data = []
        
        if st.session_state.usage_data:
            # Create a simple usage chart
            import plotly.express as px
            import pandas as pd
            
            df = pd.DataFrame(st.session_state.usage_data)
            fig = px.line(df, x='timestamp', y='searches', title='Search Activity Over Time')
            st.plotly_chart(fig, use_container_width=True)

def create_research_insights():
    """Create insights panel for research patterns."""
    st.markdown("## 🧠 Research Insights")
    
    insight_type = st.selectbox(
        "Choose insight type:",
        ["Trend Analysis", "Topic Clustering", "Citation Patterns", "Research Gaps"]
    )
    
    if insight_type == "Trend Analysis":
        st.info("🔍 Use the trend analysis tool to identify emerging research patterns.")
        topic = st.text_input("Enter research topic for trend analysis:")
        if topic and st.button("Analyze Trends"):
            with st.spinner("Analyzing trends..."):
                from agent_runner import analyze_trends
                result = analyze_trends(topic)
                st.markdown(result)
    
    elif insight_type == "Topic Clustering":
        st.info("📊 This would group related research topics together.")
        st.write("Feature coming soon...")
    
    elif insight_type == "Citation Patterns":
        st.info("📚 This would analyze citation networks and influential papers.")
        st.write("Feature coming soon...")
    
    elif insight_type == "Research Gaps":
        st.info("🎯 This would identify areas needing more research attention.")
        st.write("Feature coming soon...")

def create_advanced_search():
    """Create advanced search interface with filters."""
    st.markdown("## 🔍 Advanced Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        query = st.text_input("Search Query:")
        date_from = st.date_input("From Date:", value=None)
        date_to = st.date_input("To Date:", value=None)
    
    with col2:
        category = st.selectbox(
            "Research Category:",
            ["All", "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML"]
        )
        min_relevance = st.slider("Minimum Relevance Score:", 0.0, 1.0, 0.7)
    
    if st.button("🔍 Advanced Search") and query:
        with st.spinner("Performing advanced search..."):
            # This would integrate with the enhanced search function
            st.info("Advanced search with filters coming soon...")
            st.write(f"Searching for: {query}")
            st.write(f"Date range: {date_from} to {date_to}")
            st.write(f"Category: {category}")
            st.write(f"Min relevance: {min_relevance}")

def display_query_info(query_type: str):
    """
    Display information about how the query is being processed.
    """
    query_type_info = {
        "general": "🔍 General Research Query",
        "comparative": "⚖️ Comparative Analysis",
        "trend": "📈 Trend Analysis", 
        "technical": "🔧 Technical Deep-dive",
        "review": "📚 Literature Review",
        "implementation": "💻 Implementation Focus"
    }
    
    enhancement_info = {
        "general": "Focusing on recent research papers and academic sources",
        "comparative": "Providing detailed comparisons with specific examples",
        "trend": "Emphasizing temporal trends and recent developments",
        "technical": "Including mathematical details and technical analysis",
        "review": "Providing comprehensive literature review",
        "implementation": "Focusing on practical implementation details"
    }
    
    if query_type in query_type_info:
        st.info(f"{query_type_info[query_type]}: {enhancement_info[query_type]}") 
