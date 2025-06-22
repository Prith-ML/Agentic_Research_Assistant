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
from datetime import datetime
import re

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
    index1 = pc.Index("ragas1")  # Second index
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

# Global variables for storing search results
last_search_result = None

def search_databases_wrapper(query: str) -> str:
    """
    Wrapper function for search_databases that stores the result globally.
    This allows us to access sources later without re-running the search.
    """
    global last_search_result
    result = search_databases(query)
    last_search_result = result  # Store for later use
    return result["content"]

def classify_database(query: str) -> str:
    """
    Use LLM to intelligently classify which database to search based on the user's query.
    
    Args:
        query (str): The search query
        
    Returns:
        str: Database name to search ("ragas" or "ragas1")
    """
    classification_prompt = f"""
    Analyze this query and determine which database would be most appropriate to search:

    Query: "{query}"

    Available databases:
    - RAGAS: Contains academic research papers from arXiv (scientific studies, theoretical work, methodology, mathematical analysis, research findings)
    - RAGAS1: Contains AI tech articles, news, industry updates, company announcements, product releases, market trends, current developments

    Consider:
    - Is the user asking about academic research or industry news?
    - Are they looking for scientific papers or current developments?
    - Do they want theoretical analysis or practical updates?
    - Is this about research methodology or business/technology news?

    Respond with exactly one word: RAGAS or RAGAS1
    """

    try:
        response = llm.invoke(classification_prompt)
        database_choice = response.content.strip().upper()
        
        # Validate response
        if database_choice in ["RAGAS", "RAGAS1"]:
            logger.info(f"LLM selected database: {database_choice.lower()} for query: {query}")
            return database_choice.lower()
        else:
            logger.warning(f"Invalid LLM response: {database_choice}, defaulting to ragas")
            return "ragas"
            
    except Exception as e:
        logger.error(f"LLM classification failed: {e}, defaulting to ragas")
        return "ragas"

def search_databases(query: str) -> dict:
    """
    Enhanced search through research papers and tech articles using vector similarity with intelligent database routing.
    
    Args:
        query (str): The search query
        
    Returns:
        dict: Dictionary containing search results and source information from either research papers or tech articles
    """
    try:
        logger.info(f"Searching for: {query}")
        
        # Determine which database to search
        database_choice = classify_database(query)
        logger.info(f"Selected database: {database_choice}")
        
        # Select the appropriate index
        search_index = index1 if database_choice == "ragas1" else index
        
        xq = embed.embed_query(query)
        
        # Enhanced query with better parameters
        out = search_index.query(
            vector=xq, 
            top_k=8,  # Increased for better coverage
            include_metadata=True,
            include_values=False
        )
        
        if not out["matches"]:
            content_type = "tech articles" if database_choice == "ragas1" else "research papers"
            return {
                "content": f"Found {content_type} in {database_choice} database but none met the relevance threshold. Try rephrasing your question.",
                "sources": [],
                "paper_count": 0,
                "database_used": database_choice
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
                    # Clean the text content to remove paper headers and formatting
                    raw_text = match["metadata"]["text"]
                    
                    # Remove common paper section headers more aggressively
                    cleaned_text = raw_text
                    headers_to_remove = [
                        "Abstract", "ABSTRACT", "Introduction", "INTRODUCTION",
                        "Abstract:", "ABSTRACT:", "Introduction:", "INTRODUCTION:",
                        "1. Introduction", "1. INTRODUCTION", "1 Introduction", "1 INTRODUCTION",
                        "2. Related Work", "2. RELATED WORK", "2 Related Work", "2 RELATED WORK",
                        "3. Methodology", "3. METHODOLOGY", "3 Methodology", "3 METHODOLOGY",
                        "4. Results", "4. RESULTS", "4 Results", "4 RESULTS",
                        "5. Conclusion", "5. CONCLUSION", "5 Conclusion", "5 CONCLUSION",
                        "Abstract\n", "ABSTRACT\n", "Introduction\n", "INTRODUCTION\n",
                        "Abstract:\n", "ABSTRACT:\n", "Introduction:\n", "INTRODUCTION:\n"
                    ]
                    
                    # More aggressive cleaning - remove headers and everything before them
                    for header in headers_to_remove:
                        if header in cleaned_text:
                            # Split and take everything after the header
                            parts = cleaned_text.split(header, 1)
                            if len(parts) > 1:
                                cleaned_text = parts[1].strip()
                    
                    # Remove any remaining section numbers and headers
                    # Remove lines that are just section headers (like "1. Introduction")
                    cleaned_text = re.sub(r'^\d+\.\s*[A-Z][a-z\s]+$', '', cleaned_text, flags=re.MULTILINE)
                    # Remove bold headers (like **Abstract**)
                    cleaned_text = re.sub(r'\*\*[A-Za-z\s]+\*\*', '', cleaned_text)
                    # Remove standalone headers
                    cleaned_text = re.sub(r'^[A-Z][A-Z\s]+$', '', cleaned_text, flags=re.MULTILINE)
                    
                    # Clean up extra whitespace and formatting
                    cleaned_text = " ".join(cleaned_text.split())  # Normalize whitespace
                    
                    # Limit excerpt length for display
                    display_text = cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
                    
                    result_text = f"[Score: {score:.2f}] {title} by {authors} ({date})\n{display_text}"
                    results.append(result_text)
                    
                    # Add source information (keep original text for excerpt)
                    source_info = {
                        "title": title,
                        "authors": authors,
                        "date": date,
                        "paper_id": paper_id,
                        "url": url,
                        "relevance_score": score,
                        "excerpt": match["metadata"]["text"][:200] + "..." if len(match["metadata"]["text"]) > 200 else match["metadata"]["text"],
                        "database": database_choice
                    }
                    sources.append(source_info)
        
        if not results:
            return {
                "content": f"Found papers in {database_choice} database but none met the relevance threshold. Try rephrasing your question.",
                "sources": [],
                "paper_count": 0,
                "database_used": database_choice
            }
        
        return {
            "content": "\n---\n".join(results[:5]),  # Return top 5 most relevant
            "sources": sources[:5],  # Top 5 sources
            "paper_count": len(sources),
            "database_used": database_choice
        }
        
    except Exception as e:
        logger.error(f"Error in search_databases: {e}")
        return {
            "content": f"Error searching content: {str(e)}",
            "sources": [],
            "paper_count": 0,
            "database_used": "unknown"
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
    
    # Group sources by database
    ragas_sources = [s for s in sources if s.get('database') == 'ragas']
    ragas1_sources = [s for s in sources if s.get('database') == 'ragas1']
    
    citations = "\n\n## 📚 Sources & References\n\n"
    
    # Determine content type based on database
    if ragas_sources and ragas1_sources:
        citations += f"*Based on analysis of {len(ragas_sources)} research papers and {len(ragas1_sources)} tech articles:*\n\n"
    elif ragas_sources:
        citations += f"*Based on analysis of {len(ragas_sources)} research papers:*\n\n"
    elif ragas1_sources:
        citations += f"*Based on analysis of {len(ragas1_sources)} tech articles:*\n\n"
    else:
        citations += f"*Based on analysis of {len(sources)} sources:*\n\n"
    
    # Add database information if using multiple databases
    if ragas_sources and ragas1_sources:
        citations += "**📄 Research Papers (arXiv):**\n"
        for i, source in enumerate(ragas_sources, 1):
            citations += f"**{i}.** {source['title']}\n"
            citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
            if source.get('url'):
                citations += f"   - **Link:** [View Paper]({source['url']})\n"
            citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
        
        citations += "**📰 Tech Articles:**\n"
        for i, source in enumerate(ragas1_sources, 1):
            citations += f"**{i}.** {source['title']}\n"
            citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
            if source.get('url'):
                citations += f"   - **Link:** [View Article]({source['url']})\n"
            citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
    else:
        # Single database format
        database_type = "research papers" if ragas_sources else "tech articles"
        link_text = "View Paper" if ragas_sources else "View Article"
        
        for i, source in enumerate(sources, 1):
            citations += f"**{i}.** {source['title']}\n"
            citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
            if source.get('url'):
                citations += f"   - **Link:** [{link_text}]({source['url']})\n"
            citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
    
    # Add appropriate footer based on database used
    if ragas_sources and ragas1_sources:
        citations += "---\n*These sources were retrieved using semantic search through arXiv research papers and AI tech articles.*"
    elif ragas_sources:
        citations += "---\n*These sources were retrieved using semantic search through arXiv research papers.*"
    elif ragas1_sources:
        citations += "---\n*These sources were retrieved using semantic search through AI tech articles.*"
    else:
        citations += "---\n*These sources were retrieved using semantic search.*"
    
    return citations

def summarize_papers(query: str) -> str:
    """
    Get a summary of papers related to a topic.
    """
    try:
        # Get papers first
        search_result = search_databases(query)
        if search_result["paper_count"] == 0:
            return f"No papers found for {query}"
        
        # Simple summary prompt
        summary_prompt = f"""
        Based on the following research papers, provide a comprehensive summary of: {query}
        
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
        search_result = search_databases(recent_query)
        
        if search_result["paper_count"] == 0:
            return f"No recent papers found for {topic}"
        
        # Simple trend analysis prompt
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
        func=search_databases_wrapper,  # Use wrapper instead of lambda
        name="search_databases",
        description="Use this tool for ANY question about AI companies, industry developments, product announcements, company research, current AI news, AI models, AI developments, or any specific AI-related information. "
                   "This tool searches through a curated dataset of scientific papers and tech articles "
                   "and returns relevant excerpts to help answer questions about current AI developments."
    ),
    Tool.from_function(
        func=summarize_papers,
        name="summarize_papers",
        description="Use this tool when the user explicitly asks for a 'summary', 'overview', 'literature review', or 'comprehensive analysis' of research papers on a topic. "
                   "This tool will: 1) Search for relevant papers, 2) Provide a structured summary with key findings, methodologies, challenges, and future directions, "
                   "3) Include proper citations. Use this for academic writing, literature reviews, or when users want a complete research overview."
    ),
    Tool.from_function(
        func=analyze_trends,
        name="analyze_trends",
        description="Use this tool when the user explicitly asks for 'trends', 'emerging directions', 'recent developments', 'evolution', 'future directions', 'industry insights', or 'company developments' in a research area or AI company. "
                   "This tool will: 1) Search for recent papers and developments, 2) Provide a structured trend analysis with emerging patterns, methodologies, research shifts, and breakthroughs, "
                   "3) Include proper citations. Use this for understanding current research directions, identifying emerging areas, analyzing how a field is evolving, or getting company insights."
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
        | prompt.partial(
            tools=convert_tools(tools),
            system_message="CRITICAL: For questions about AI companies, industry developments, current AI news, product announcements, company research, or any specific AI-related information, you MUST use the search tools to get current and accurate information. "
                          "For questions about AI models, AI developments, or any AI topic that might have recent updates, ALWAYS search the database first. "
                          "Use search_databases for company information and current developments, summarize_papers for research summaries, or analyze_trends for trend analysis. "
                          "Only use your knowledge for basic definitions that don't require current information. "
                          "This is a research assistant - your primary job is to search databases and provide sourced information."
        )
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

# Global chat history and cache with better memory management
chat_history = []
response_cache = {}
MAX_HISTORY_EXCHANGES = 6  # 3 exchanges (6 messages: 3 human + 3 assistant)
MAX_CACHE_SIZE = 100
CACHE_CLEANUP_SIZE = 20

# Query enhancement and classification system
QUERY_ENHANCEMENTS = {
    "company_info": "Focus on current company developments, industry news, and recent announcements from AI companies.",
    "general": "Focus on recent research papers and academic sources.",
    "comparative": "Provide detailed comparisons with specific examples from research papers.",
    "trend": "Emphasize temporal trends and evolution of the field with recent developments.",
    "technical": "Include mathematical details, implementation specifics, and technical analysis.",
    "review": "Provide comprehensive literature review with key findings and research gaps.",
    "implementation": "Focus on practical implementation details and code considerations."
}

def classify_query_type(query: str) -> str:
    """
    Classify query type for optimal prompt selection.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: Query type classification
    """
    query_lower = query.lower()
    
    # Check for company/industry queries (should trigger database search)
    if any(word in query_lower for word in ["company", "industry", "insight", "development", "announcement", "product", "news", "current", "latest", "recent"]):
        return "company_info"
    
    # Check for comparative queries
    elif any(word in query_lower for word in ["compare", "difference", "vs", "versus", "versus", "contrast", "similarity"]):
        return "comparative"
    
    # Check for trend queries
    elif any(word in query_lower for word in ["trend", "evolution", "development", "latest", "recent", "emerging", "future"]):
        return "trend"
    
    # Check for technical queries
    elif any(word in query_lower for word in ["how", "implement", "technique", "algorithm", "method", "approach", "architecture"]):
        return "technical"
    
    # Check for review queries
    elif any(word in query_lower for word in ["review", "summary", "overview", "survey", "literature"]):
        return "review"
    
    # Check for implementation queries
    elif any(word in query_lower for word in ["code", "implementation", "practical", "example", "tutorial"]):
        return "implementation"
    
    # Default to general
    else:
        return "general"

def enhance_query(query: str, query_type: str | None = None) -> str:
    """
    Enhance user queries with research-specific context.
    
    Args:
        query (str): The original user query
        query_type (str | None): The classified query type (auto-detected if None)
        
    Returns:
        str: Enhanced query with research context
    """
    if query_type is None:
        query_type = classify_query_type(query)
    
    enhancement = QUERY_ENHANCEMENTS.get(query_type, QUERY_ENHANCEMENTS["general"])
    enhanced_query = f"{query} {enhancement}"
    
    logger.info(f"Query enhanced: '{query}' -> Type: {query_type}")
    return enhanced_query

def chat(query: str) -> str:
    """
    Enhanced chat function that respects agent's tool choice and uses database as fallback.
    
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
        
        # Enhance the query for better results
        query_type = classify_query_type(query)
        enhanced_query = enhance_query(query, query_type)
        
        # Execute the agent with enhanced query
        out = agent_executor.invoke({
            "input": enhanced_query,
            "chat_history": ""
        })
        
        agent_answer = out.get("output", "")
        
        # Check if agent used any tools
        agent_used_tools = False
        sources_section = ""
        
        try:
            if "intermediate_steps" in out:
                for step in out["intermediate_steps"]:
                    logger.info(f"Agent used tool: {step[0].tool}")
                    agent_used_tools = True
                    
                    # Check for search_databases tool
                    if step[0].tool == "search_databases" and last_search_result and last_search_result["sources"]:
                        sources_section = format_sources(last_search_result["sources"])
                        break
                    
                    # Check for summarize_papers tool - extract sources from the response
                    elif step[0].tool == "summarize_papers":
                        logger.info("Agent used summarize_papers tool")
                        # The summarize_papers function already includes sources in its response
                        agent_output = step[1] if len(step) > 1 else ""
                        if "## 📚 Sources & References" in agent_output:
                            # Extract the sources section from the agent's response
                            sources_start = agent_output.find("## 📚 Sources & References")
                            sources_section = agent_output[sources_start:]
                            break
                    
                    # Check for analyze_trends tool - extract sources from the response
                    elif step[0].tool == "analyze_trends":
                        logger.info("Agent used analyze_trends tool")
                        # The analyze_trends function already includes sources in its response
                        agent_output = step[1] if len(step) > 1 else ""
                        if "## 📚 Sources & References" in agent_output:
                            # Extract the sources section from the agent's response
                            sources_start = agent_output.find("## 📚 Sources & References")
                            sources_section = agent_output[sources_start:]
                            break
                            
        except Exception as e:
            logger.warning(f"Could not extract sources from agent: {e}")
        
        # If agent didn't use tools, try database search as fallback
        if not agent_used_tools:
            logger.info("Agent didn't use tools, trying database search as fallback")
            search_result = search_databases(query)
            if search_result["paper_count"] > 0:
                logger.info(f"Fallback search found {search_result['paper_count']} sources")
                sources_section = format_sources(search_result["sources"])
                # Replace agent's short answer with database content + sources
                full_response = search_result["content"] + sources_section
            else:
                full_response = agent_answer
        else:
            # Agent used tools, use its response
            full_response = agent_answer
        
        # Cache the response
        response_cache[cache_key] = full_response
        
        # Improved cache management
        if len(response_cache) > MAX_CACHE_SIZE:
            oldest_keys = list(response_cache.keys())[:CACHE_CLEANUP_SIZE]
            for key in oldest_keys:
                del response_cache[key]
        
        # Update chat history
        exchange = {
            "human": query,
            "assistant": agent_answer,  # Store agent response without sources
            "timestamp": datetime.now(),
            "query_type": query_type
        }
        chat_history.append(exchange)
        
        if len(chat_history) > MAX_HISTORY_EXCHANGES:
            chat_history = chat_history[-MAX_HISTORY_EXCHANGES:]
        
        logger.info(f"Query processed successfully. Type: {query_type}, Agent used tools: {agent_used_tools}")
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

def clear_history():
    """Clear the chat history."""
    global chat_history
    chat_history.clear()
    logger.info("Chat history cleared")

def get_chat_stats():
    """Get statistics about the chat session."""
    global chat_history, response_cache
    return {
        "cache_size": len(response_cache),
        "history_exchanges": len(chat_history),
        "memory_usage_mb": estimate_memory_usage(),
        "avg_exchange_length": calculate_avg_exchange_length()
    }

def estimate_memory_usage():
    """Estimate memory usage in MB."""
    import sys
    
    # Estimate memory for chat history
    history_memory = sum(
        sys.getsizeof(exchange["human"]) + sys.getsizeof(exchange["assistant"])
        for exchange in chat_history
    )
    
    # Estimate memory for cache
    cache_memory = sum(
        sys.getsizeof(key) + sys.getsizeof(value)
        for key, value in response_cache.items()
    )
    
    total_memory = history_memory + cache_memory
    return round(total_memory / (1024 * 1024), 2)  # Convert to MB

def calculate_avg_exchange_length():
    """Calculate average length of exchanges."""
    if not chat_history:
        return 0
    
    total_length = sum(
        len(exchange["human"]) + len(exchange["assistant"])
        for exchange in chat_history
    )
    return round(total_length / len(chat_history), 0)

def test_source_extraction():
    """Test function to verify source extraction is working."""
    test_query = "What are transformer architectures?"
    logger.info(f"Testing source extraction with query: {test_query}")
    
    # Test direct search
    search_result = search_databases(test_query)
    logger.info(f"Search result: {search_result['paper_count']} papers found")
    
    if search_result["sources"]:
        logger.info(f"Sources found: {len(search_result['sources'])}")
        for i, source in enumerate(search_result["sources"][:2]):  # Show first 2
            logger.info(f"Source {i+1}: {source['title']} (Score: {source['relevance_score']:.2f})")
        
        # Test formatting
        formatted = format_sources(search_result["sources"])
        logger.info(f"Formatted sources length: {len(formatted)} characters")
        return True
    else:
        logger.error("No sources found in test search")
        return False
