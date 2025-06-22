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

def display_agentic_stats():
    """Display agentic AI statistics."""
    try:
        from agent_runner import get_agentic_insights
        insights = get_agentic_insights()
        
        st.markdown("### 🤖 Agentic AI Stats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            research_plans = insights.get("research_plans_created", 0)
            st.metric(
                "Research Plans Created",
                research_plans,
                help="Number of complex goals that required planning"
            )
        
        with col2:
            proactive_suggestions = insights.get("proactive_suggestions", 0)
            st.metric(
                "Proactive Suggestions",
                proactive_suggestions,
                help="Number of follow-up questions and suggestions generated"
            )
        
        with col3:
            tool_effectiveness = insights.get("tool_effectiveness", {})
            total_tool_uses = sum(stats.get("total_uses", 0) for stats in tool_effectiveness.values())
            st.metric(
                "Tool Uses",
                total_tool_uses,
                help="Total number of tool executions"
            )
        
        # Display tool effectiveness
        if tool_effectiveness:
            st.markdown("#### 🔧 Tool Effectiveness")
            for tool_name, stats in tool_effectiveness.items():
                if stats.get("total_uses", 0) > 0:
                    success_rate = (stats.get("successful_uses", 0) / stats.get("total_uses", 1)) * 100
                    avg_quality = stats.get("avg_quality", 0) * 100
                    
                    st.write(f"**{tool_name}**: {success_rate:.1f}% success rate, {avg_quality:.1f}% avg quality")
        
    except Exception as e:
        st.info("Agentic stats will appear after first interaction")

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
                "Number of sources to search", 
                min_value=1, 
                max_value=10, 
                value=5,
                help="Number of relevant sources to include in search results"
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
            
            show_database_info = st.checkbox(
                "Show database selection info",
                value=True,
                help="Display which database the AI selected for each query"
            )
        
        return {
            "search_results": search_results,
            "temperature": temperature,
            "show_timestamps": show_timestamps,
            "auto_scroll": auto_scroll,
            "show_database_info": show_database_info
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
        "🔬 **Technical details**: Ask for explanations of specific algorithms or concepts",
        "🤖 **AI Database Selection**: The AI automatically chooses the best database (academic papers, industry articles, or both) based on your query",
        "📄 **Academic queries**: Ask about research methodologies, theoretical concepts, or scientific studies",
        "🚀 **Industry queries**: Ask about practical applications, company announcements, or implementation guides"
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

def create_agentic_welcome_section():
    """Create an enhanced welcome section highlighting agentic features."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;'>
        <h1>🤖 Agentic Research Assistant</h1>
        <p style='font-size: 1.2rem; margin: 1rem 0;'>Your intelligent research companion with autonomous planning and self-reflection</p>
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <strong>Agentic Features:</strong> Goal Planning • Self-Reflection • Proactive Suggestions • Tool Optimization
        </div>
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <strong>🤖 Intelligent Database Selection:</strong> AI automatically chooses the best database (Academic Papers • Industry Articles • Both) based on your query
        </div>
        <div style='margin-top: 1rem; font-size: 0.9rem;'>
            <strong>Try asking:</strong> "Give me a comprehensive analysis of transformer architectures" or "Research the latest developments in few-shot learning"
        </div>
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

def create_agentic_insights_panel():
    """Create a panel showing agentic AI insights and performance."""
    st.markdown("## 🤖 Agentic AI Insights")
    
    try:
        from agent_runner import get_agentic_insights
        insights = get_agentic_insights()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Performance Metrics")
            
            # Quality trends
            quality_trends = insights.get("quality_trends", [])
            if quality_trends:
                recent_quality = quality_trends[-1]["reflection"].quality_score if quality_trends else 0
                st.metric("Recent Response Quality", f"{recent_quality:.1%}")
            
            # Research plans
            research_plans = insights.get("research_plans_created", 0)
            st.metric("Complex Goals Handled", research_plans)
            
            # Proactive suggestions
            proactive_count = insights.get("proactive_suggestions", 0)
            st.metric("Proactive Suggestions", proactive_count)
        
        with col2:
            st.markdown("### 🔧 Tool Performance")
            
            tool_effectiveness = insights.get("tool_effectiveness", {})
            if tool_effectiveness:
                for tool_name, stats in tool_effectiveness.items():
                    if stats.get("total_uses", 0) > 0:
                        success_rate = (stats.get("successful_uses", 0) / stats.get("total_uses", 1)) * 100
                        avg_quality = stats.get("avg_quality", 0) * 100
                        
                        st.write(f"**{tool_name}**")
                        st.write(f"Success: {success_rate:.1f}% | Quality: {avg_quality:.1f}%")
                        st.progress(success_rate / 100)
        
        # Learning insights
        st.markdown("### 🧠 Learning Insights")
        
        if quality_trends:
            st.write("**Recent Quality Trends:**")
            for trend in quality_trends[-3:]:  # Last 3
                quality = trend["reflection"].quality_score
                timestamp = trend["timestamp"].strftime("%H:%M")
                st.write(f"• {timestamp}: {quality:.1%} quality")
        
        # Agentic behavior summary
        st.markdown("### 🎯 Agentic Behavior Summary")
        
        agentic_behaviors = []
        if research_plans > 0:
            agentic_behaviors.append("✅ Goal-oriented planning")
        if proactive_count > 0:
            agentic_behaviors.append("✅ Proactive suggestions")
        if any(stats.get("avg_quality", 0) > 0.8 for stats in tool_effectiveness.values()):
            agentic_behaviors.append("✅ High-quality responses")
        
        if agentic_behaviors:
            for behavior in agentic_behaviors:
                st.write(behavior)
        else:
            st.write("🔄 Agentic features will activate with more interactions")
    
    except Exception as e:
        st.info("Agentic insights will appear after first interaction")

def create_research_planning_interface():
    """Create an interface for research planning and goal setting."""
    st.markdown("## 🎯 Research Planning")
    
    st.info("""
    **Agentic Planning Features:**
    - **Goal Decomposition**: Complex research goals are broken down into manageable sub-tasks
    - **Multi-step Execution**: Systematic approach to research with progress tracking
    - **Tool Optimization**: Automatic selection of best tools for each sub-task
    - **Quality Assurance**: Self-evaluation and improvement suggestions
    """)
    
    # Example research goals
    st.markdown("### 💡 Example Complex Research Goals")
    
    example_goals = [
        "Give me a comprehensive analysis of transformer architectures in natural language processing",
        "Research the latest developments in few-shot learning and their applications",
        "Compare different approaches to reinforcement learning in robotics",
        "Investigate the current state of research in computer vision for medical imaging",
        "Analyze trends in machine learning for climate change prediction"
    ]
    
    selected_goal = st.selectbox(
        "Try a complex research goal:",
        ["Select a goal..."] + example_goals
    )
    
    if selected_goal and selected_goal != "Select a goal...":
        st.write(f"**Selected Goal:** {selected_goal}")
        st.write("This will trigger the agentic planning system with:")
        st.write("• Goal decomposition into sub-tasks")
        st.write("• Multi-step research execution")
        st.write("• Self-evaluation and improvement")
        st.write("• Proactive follow-up suggestions")

def create_agentic_features_guide():
    """Create a guide explaining agentic AI features."""
    st.markdown("## 🤖 Agentic AI Features Guide")
    
    features = [
        {
            "name": "🎯 Goal-Oriented Planning",
            "description": "Automatically breaks down complex research goals into manageable sub-tasks",
            "example": "Ask: 'Give me a comprehensive analysis of transformer architectures'"
        },
        {
            "name": "🤔 Self-Reflection",
            "description": "Evaluates response quality and identifies areas for improvement",
            "example": "Shows confidence levels and quality scores for each response"
        },
        {
            "name": "🔍 Gap Detection",
            "description": "Identifies missing information and suggests additional research",
            "example": "Highlights topics that need more exploration"
        },
        {
            "name": "💡 Proactive Suggestions",
            "description": "Generates follow-up questions and research directions",
            "example": "Suggests related topics and future research areas"
        },
        {
            "name": "🔧 Tool Optimization",
            "description": "Learns which tools work best for different types of queries",
            "example": "Tracks tool effectiveness and improves over time"
        }
    ]
    
    for feature in features:
        with st.expander(f"**{feature['name']}**"):
            st.write(feature['description'])
            st.write(f"**Example:** {feature['example']}") 
