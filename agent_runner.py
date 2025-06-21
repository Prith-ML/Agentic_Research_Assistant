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

def arxiv_search(query: str) -> dict:
    """
    Enhanced search through arXiv papers using vector similarity.
    
    Args:
        query (str): The search query
        
    Returns:
        dict: Dictionary containing search results and source information
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
            return {
                "content": "No relevant papers found for this query.",
                "sources": [],
                "paper_count": 0
            }
        
        # Enhanced result processing with ranking and source tracking
        results = []
        sources = []
        
        for match in out["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                score = match.get("score", 0)
                title = match["metadata"].get("title", "Unknown Title")
                authors = match["metadata"].get("authors", "Unknown Authors")
                date = match["metadata"].get("date", "Unknown Date")
                paper_id = match["metadata"].get("paper_id", "Unknown ID")
                url = match["metadata"].get("url", "")
                
                # Only include high-quality matches
                if score > 0.7:  # Threshold for relevance
                    result_text = f"[Score: {score:.2f}] {title} by {authors} ({date})\n{match['metadata']['text']}"
                    results.append(result_text)
                    
                    # Add source information
                    source_info = {
                        "title": title,
                        "authors": authors,
                        "date": date,
                        "paper_id": paper_id,
                        "url": url,
                        "relevance_score": score,
                        "excerpt": match["metadata"]["text"][:200] + "..." if len(match["metadata"]["text"]) > 200 else match["metadata"]["text"]
                    }
                    sources.append(source_info)
        
        if not results:
            return {
                "content": "Found papers but none met the relevance threshold. Try rephrasing your question.",
                "sources": [],
                "paper_count": 0
            }
        
        return {
            "content": "\n---\n".join(results[:5]),  # Return top 5 most relevant
            "sources": sources[:5],  # Top 5 sources
            "paper_count": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Error in arxiv_search: {e}")
        return {
            "content": f"Error searching papers: {str(e)}",
            "sources": [],
            "paper_count": 0
        }

def format_sources(sources: list) -> str:
    """
    Format sources into a nice citation format.
    
    Args:
        sources (list): List of source dictionaries
        
    Returns:
        str: Formatted citations
    """
    if not sources:
        return ""
    
    citations = "\n\n## 📚 Sources & References\n\n"
    citations += f"*Based on analysis of {len(sources)} research papers:*\n\n"
    
    for i, source in enumerate(sources, 1):
        citations += f"**{i}.** {source['title']}\n"
        citations += f"   - **Authors:** {source['authors']}\n"
        citations += f"   - **Date:** {source['date']}\n"
        citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
        if source.get('url'):
            citations += f"   - **Link:** [View Paper]({source['url']})\n"
        citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
    
    citations += "---\n*These sources were retrieved using semantic search through arXiv papers.*"
    return citations

def semantic_search(query: str, context: str = "") -> dict:
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
        search_result = arxiv_search(query)
        if search_result["paper_count"] == 0:
            return search_result["content"]
        
        # Ask the LLM to summarize
        summary_prompt = f"""
        Based on the following research papers, provide a comprehensive summary of the current state of research on: {query}
        
        Papers:
        {search_result["content"]}
        
        Please provide:
        1. Key findings and trends
        2. Main methodologies used
        3. Current challenges and limitations
        4. Future research directions
        """
        
        response = llm.invoke(summary_prompt)
        summary = str(response.content)
        
        # Add sources to the summary
        sources_section = format_sources(search_result["sources"])
        return summary + sources_section
        
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
        search_result = arxiv_search(recent_query)
        
        if search_result["paper_count"] == 0:
            return f"No recent papers found for {topic}"
        
        # Analyze trends
        trend_prompt = f"""
        Analyze the following recent papers to identify trends in {topic}:
        
        {search_result["content"]}
        
        Please identify:
        1. Emerging trends and patterns
        2. New methodologies being adopted
        3. Shifts in research focus
        4. Key breakthroughs or innovations
        5. Areas gaining more attention
        """
        
        response = llm.invoke(trend_prompt)
        analysis = str(response.content)
        
        # Add sources to the analysis
        sources_section = format_sources(search_result["sources"])
        return analysis + sources_section
        
    except Exception as e:
        logger.error(f"Error in analyze_trends: {e}")
        return f"Error analyzing trends: {str(e)}"

# Register enhanced tools for the agent
tools = [
    Tool.from_function(
        func=lambda query: arxiv_search(query)["content"],  # Extract content for tool
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
    Enhanced chat function with caching, better memory management, and source citations.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The AI assistant's response with sources
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
        
        # Try to get sources from the search results
        sources_section = ""
        try:
            # If the agent used arxiv_search, we can extract sources
            if "intermediate_steps" in out:
                for step in out["intermediate_steps"]:
                    if step[0].tool == "arxiv_search":
                        # Extract sources from the search
                        search_result = arxiv_search(step[0].tool_input)
                        if search_result["sources"]:
                            sources_section = format_sources(search_result["sources"])
                            break
        except Exception as e:
            logger.warning(f"Could not extract sources: {e}")
        
        # Combine answer with sources
        full_response = answer
        if sources_section:
            full_response += sources_section
        
        # Cache the response
        response_cache[cache_key] = full_response
        
        # Limit cache size to prevent memory issues
        if len(response_cache) > 100:
            # Remove oldest entries
            oldest_keys = list(response_cache.keys())[:20]
            for key in oldest_keys:
                del response_cache[key]
        
        # Update chat history (keep last 10 exchanges to prevent it from growing too large)
        chat_history += f"\nHuman: {query}\nAssistant: {answer}"  # Store without sources in history
        history_lines = chat_history.split('\n')
        if len(history_lines) > 20:  # Keep only last 10 exchanges
            chat_history = '\n'.join(history_lines[-20:])
        
        logger.info("Query processed successfully")
        return full_response
        
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
