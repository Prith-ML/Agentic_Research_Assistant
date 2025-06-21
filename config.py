"""
Configuration settings for the AI Research Assistant.
"""

# App Configuration
APP_CONFIG = {
    "title": "AI Research Assistant",
    "icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model Configuration
MODEL_CONFIG = {
    "model_name": "claude-3-5-haiku-20241022",
    "temperature": 0.2,
    "max_tokens": 4000
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_id": "cohere.embed-english-v3",
    "region_name": "us-east-1"
}

# Pinecone Configuration
PINECONE_CONFIG = {
    "index_name": "ragas",  # Replace with your actual index name
    "top_k": 5,
    "include_metadata": True
}

# UI Configuration
UI_CONFIG = {
    "default_search_results": 5,
    "default_temperature": 0.2,
    "show_timestamps": False,
    "auto_scroll": True,
    "chat_input_placeholder": "Ask a question about AI, ML, or research papers..."
}

# Quick Actions
QUICK_ACTIONS = [
    {
        "label": "🤖 AI Trends",
        "query": "What are the latest trends in artificial intelligence research?"
    },
    {
        "label": "🧠 ML Methods", 
        "query": "Explain the key machine learning methods and their applications"
    },
    {
        "label": "📊 Data Science",
        "query": "What are the recent advances in data science and analytics?"
    },
    {
        "label": "🔍 Computer Vision",
        "query": "What are the latest developments in computer vision?"
    },
    {
        "label": "📝 NLP Advances",
        "query": "What are the recent advances in natural language processing?"
    },
    {
        "label": "🎯 Research Tips",
        "query": "How can I effectively conduct literature reviews in AI/ML?"
    }
]

# Research Tips
RESEARCH_TIPS = [
    "🔍 **Be specific**: Instead of 'AI', ask about 'transformer architectures' or 'few-shot learning'",
    "📚 **Ask for summaries**: Request paper summaries or literature reviews on specific topics",
    "🎯 **Compare methods**: Ask to compare different approaches or techniques",
    "📊 **Recent trends**: Inquire about the latest developments in your field of interest",
    "🔬 **Technical details**: Ask for explanations of specific algorithms or concepts"
]

# CSS Styles
CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #764ba2;
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .clear-button {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.9rem;
    }
    
    .clear-button:hover {
        opacity: 0.9;
    }
    
    .quick-action-btn {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
"""

# Error Messages
ERROR_MESSAGES = {
    "api_key_missing": "API key missing. Please check your configuration.",
    "initialization_failed": "Failed to initialize component: {component}",
    "search_failed": "Error searching papers: {error}",
    "chat_failed": "I apologize, but I encountered an error while processing your query: {error}",
    "no_response": "I apologize, but I couldn't generate a response for your query.",
    "no_papers_found": "No relevant papers found for this query."
}

# Success Messages
SUCCESS_MESSAGES = {
    "feedback_received": "Thank you for your feedback!",
    "chat_cleared": "Chat history cleared successfully.",
    "export_success": "Chat history exported successfully."
} 
