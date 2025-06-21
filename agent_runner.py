#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import time
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor
from langchain.agents.output_parsers.xml import XMLAgentOutputParser
from langchain import hub
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_anthropic import ChatAnthropic
from pinecone import Pinecone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (for local development)
try:
    load_dotenv()
except:
    pass

# Get API keys from Streamlit secrets
try:
    CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except Exception as e:
    logger.error(f"Failed to load secrets: {e}")
    raise

# Validate API keys
assert CLAUDE_API_KEY is not None, "Claude API key missing."
assert PINECONE_API_KEY is not None, "Pinecone API key missing."

# Initialize LLM
try:
    llm = ChatAnthropic(
        model_name="claude-3-5-haiku-20241022",
        temperature=0.2,
        anthropic_api_key=CLAUDE_API_KEY
    )
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

# Initialize embeddings
try:
    embed = BedrockEmbeddings(
        model_id="cohere.embed-english-v3",
        region_name="us-east-1"
    )
    logger.info("Embeddings initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    raise

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("ragas")  # Replace with your index name
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

def arxiv_search(query: str) -> str:
    """
    Enhanced search through arXiv papers using vector similarity.
    
    Args:
        query (str): The search query
        
    Returns:
        str: Relevant paper excerpts joined with separators
    """
    try:
        logger.info(f"Searching for: {query}")
        xq = embed.embed_query(query)
        
        # Enhanced query with better parameters
        out = index.query(
            vector=xq, 
            top_k=8,  # Increased for better coverage
            include_metadata=True,
            include_values=False
        )
        
        if not out["matches"]:
            return "No relevant papers found for this query."
        
        # Enhanced result processing with ranking
        results = []
        for match in out["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                score = match.get("score", 0)
                title = match["metadata"].get("title", "Unknown Title")
                authors = match["metadata"].get("authors", "Unknown Authors")
                date = match["metadata"].get("date", "Unknown Date")
                
                # Only include high-quality matches
                if score > 0.7:  # Threshold for relevance
                    result_text = f"[Score: {score:.2f}] {title} by {authors} ({date})\n{match['metadata']['text']}"
                    results.append(result_text)
        
        if not results:
            return "Found papers but none met the relevance threshold. Try rephrasing your question."
        
        return "\n---\n".join(results[:5])  # Return top 5 most relevant
        
    except Exception as e:
        logger.error(f"Error in arxiv_search: {e}")
        return f"Error searching papers: {str(e)}"

def semantic_search(query: str, context: str = "") -> str:
    """
    Semantic search that considers conversation context.
    """
    # Combine query with context for better search
    enhanced_query = f"{query} {context}".strip()
    return arxiv_search(enhanced_query)

def summarize_papers(query: str) -> str:
    """
    Get a comprehensive summary of papers related to a topic.
    """
    try:
        # Get papers first
        papers = arxiv_search(query)
        if "No relevant papers found" in papers:
            return papers
        
        # Ask the LLM to summarize
        summary_prompt = f"""
        Based on the following research papers, provide a comprehensive summary of the current state of research on: {query}
        
        Papers:
        {papers}
        
        Please provide:
        1. Key findings and trends
        2. Main methodologies used
        3. Current challenges and limitations
        4. Future research directions
        """
        
        response = llm.invoke(summary_prompt)
        return str(response.content)
        
    except Exception as e:
        logger.error(f"Error in summarize_papers: {e}")
        return f"Error summarizing papers: {str(e)}"

def analyze_trends(topic: str) -> str:
    """
    Analyze trends in a specific research area.
    """
    try:
        # Search for recent papers
        recent_query = f"latest developments {topic} 2024 2023"
        papers = arxiv_search(recent_query)
        
        if "No relevant papers found" in papers:
            return f"No recent papers found for {topic}"
        
        # Analyze trends
        trend_prompt = f"""
        Analyze the following recent papers to identify trends in {topic}:
        
        {papers}
        
        Please identify:
        1. Emerging trends and patterns
        2. New methodologies being adopted
        3. Shifts in research focus
        4. Key breakthroughs or innovations
        5. Areas gaining more attention
        """
        
        response = llm.invoke(trend_prompt)
        return str(response.content)
        
    except Exception as e:
        logger.error(f"Error in analyze_trends: {e}")
        return f"Error analyzing trends: {str(e)}"

# Register enhanced tools for the agent
tools = [
    Tool.from_function(
        func=arxiv_search,
        name="arxiv_search",
        description="Use this tool to answer questions about AI, ML, or arXiv papers. "
                   "This tool searches through a curated dataset of scientific papers "
                   "and returns relevant excerpts to help answer research questions."
    ),
    Tool.from_function(
        func=summarize_papers,
        name="summarize_papers",
        description="Use this tool to get comprehensive summaries of research papers on a specific topic. "
                   "This is useful for literature reviews and understanding the current state of research."
    ),
    Tool.from_function(
        func=analyze_trends,
        name="analyze_trends",
        description="Use this tool to analyze trends and patterns in a specific research area. "
                   "This helps identify emerging directions and shifts in research focus."
    )
]

# Load XML agent prompt
try:
    prompt = hub.pull("hwchase17/xml-agent-convo")
    logger.info("Agent prompt loaded successfully")
except Exception as e:
    logger.error(f"Failed to load agent prompt: {e}")
    raise

def convert_intermediate_steps(intermediate_steps):
    """Convert intermediate steps to XML format for the agent."""
    log = ""
    for action, observation in intermediate_steps:
        log += f"<tool>{action.tool}</tool><tool_input>{action.tool_input}</tool_input>"
        log += f"<observation>{observation}</observation>"
    return log

def convert_tools(tools):
    """Convert tools to string format for the agent prompt."""
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# Initialize agent
try:
    agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", ""),
            "agent_scratchpad": lambda x: convert_intermediate_steps(x.get("intermediate_steps", [])),
        }
        | prompt.partial(tools=convert_tools(tools))
        | llm.bind(stop=["</tool_input>", "</final_answer>"])
        | XMLAgentOutputParser()
    )
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        return_intermediate_steps=True, 
        verbose=True
    )
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    raise

# Global chat history and cache
chat_history = ""
response_cache = {}

def chat(query: str) -> str:
    """
    Enhanced chat function with caching and better memory management.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The AI assistant's response
    """
    global chat_history, response_cache
    
    try:
        logger.info(f"Processing query: {query}")
        
        # Check cache for similar queries
        cache_key = query.lower().strip()
        if cache_key in response_cache:
            logger.info("Returning cached response")
            return response_cache[cache_key]
        
        # Add a small delay to prevent rate limiting
        time.sleep(1)
        
        # Extract context from recent conversation (last 3 exchanges)
        context_lines = chat_history.split('\n')[-6:]  # Last 6 lines (3 exchanges)
        recent_context = ' '.join(context_lines)
        
        # Execute the agent with context
        out = agent_executor.invoke({
            "input": query,
            "chat_history": recent_context
        })
        
        answer = out.get("output", "I apologize, but I couldn't generate a response for your query.")
        
        # Cache the response
        response_cache[cache_key] = answer
        
        # Limit cache size to prevent memory issues
        if len(response_cache) > 100:
            # Remove oldest entries
            oldest_keys = list(response_cache.keys())[:20]
            for key in oldest_keys:
                del response_cache[key]
        
        # Update chat history (keep last 10 exchanges to prevent it from growing too large)
        chat_history += f"\nHuman: {query}\nAssistant: {answer}"
        history_lines = chat_history.split('\n')
        if len(history_lines) > 20:  # Keep only last 10 exchanges
            chat_history = '\n'.join(history_lines[-20:])
        
        logger.info("Query processed successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        error_message = f"I apologize, but I encountered an error while processing your query: {str(e)}"
        return error_message

def clear_cache():
    """Clear the response cache."""
    global response_cache
    response_cache.clear()
    logger.info("Response cache cleared")

def get_chat_stats():
    """Get statistics about the chat session."""
    global chat_history, response_cache
    return {
        "cache_size": len(response_cache),
        "history_length": len(chat_history.split('\n')),
        "exchanges": len([line for line in chat_history.split('\n') if line.startswith('Human:')])
    }

# In[ ]:
