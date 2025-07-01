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
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables 
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
    # Initialize both databases
    database1_index = pc.Index("database1")  # arXiv research papers
    database2_index = pc.Index("database2")  # AI Tech articles
    logger.info("Pinecone databases initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

def select_database_for_query(query: str) -> str:
    """
    Use LLM to intelligently select the most appropriate database for a query.
    
    Args:
        query (str): The user's query
        
    Returns:
        str: Selected database ('database1', 'database2', or 'both')
    """
    try:
        selection_prompt = f"""
        Analyze this query and determine which database would be most appropriate:
        
        QUERY: "{query}"
        
        Available databases:
        1. DATABASE1: Academic research papers from arXiv (scientific papers, research methodologies, theoretical concepts, academic studies)
        2. DATABASE2: AI Tech articles (industry news, practical applications, company announcements, product releases, implementation guides)
        
        Consider:
        - Academic vs industry focus
        - Research vs practical applications
        - Theoretical vs implementation content
        - Recent developments vs established research
        
        Return ONLY one of these options:
        - "database1" (for academic/research queries)
        - "database2" (for industry/practical queries)  
        - "both" (for queries that need both academic and industry perspectives)
        
        Your response should be just the database name(s), nothing else.
        """
        
        response = llm.invoke(selection_prompt)
        selected_db = response.content.strip().lower()
        
        # Validate the response
        valid_options = ["database1", "database2", "both"]
        if selected_db not in valid_options:
            logger.warning(f"Invalid database selection: {selected_db}, defaulting to 'both'")
            return "both"
        
        logger.info(f"LLM selected database: {selected_db} for query: {query}")
        return selected_db
        
    except Exception as e:
        logger.error(f"Error in database selection: {e}")
        return "both"  # Default to both databases on error

# ============================================================================
# AGENTIC AI SYSTEM - SIMPLIFIED COMPONENTS
# ============================================================================

class ProactiveAgent:
    """Generates follow-up questions and research suggestions."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_follow_up_questions(self, query: str, response: str) -> List[str]:
        """Generate relevant follow-up questions based on the response."""
        prompt = f"""
        Based on this query and response, generate 2-3 relevant follow-up questions:
        
        QUERY: {query}
        RESPONSE: {response}
        
        Generate questions that:
        1. Explore related topics mentioned in the response
        2. Ask for more specific details or examples
        3. Request comparisons or deeper analysis
        
        Return as a simple list, one question per line.
        """
        
        try:
            response_text = self.llm.invoke(prompt).content
            questions = [q.strip() for q in response_text.split('\n') if q.strip() and '?' in q]
            return questions[:3]  # Return top 3 questions
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    def suggest_research_directions(self, query: str, response: str) -> List[str]:
        """Suggest potential research directions based on the query and response."""
        prompt = f"""
        Based on this query and response, suggest 2-3 potential research directions:
        
        QUERY: {query}
        RESPONSE: {response}
        
        Suggest directions that:
        1. Build upon the current findings
        2. Explore related areas or applications
        3. Address potential gaps or limitations
        
        Return as a simple list, one direction per line.
        """
        
        try:
            response_text = self.llm.invoke(prompt).content
            directions = [d.strip() for d in response_text.split('\n') if d.strip()]
            return directions[:3]  # Return top 3 directions
        except Exception as e:
            logger.error(f"Error suggesting research directions: {e}")
            return []

class ToolEffectivenessTracker:
    """Tracks and optimizes tool effectiveness."""
    
    def __init__(self):
        self.tool_stats = {}
    
    def record_tool_usage(self, tool_name: str, query: str, success: bool, quality_score: float = None):
        """Record tool usage and effectiveness."""
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {
                "total_uses": 0,
                "successful_uses": 0,
                "avg_quality": 0.0
            }
        
        stats = self.tool_stats[tool_name]
        stats["total_uses"] += 1
        
        if success:
            stats["successful_uses"] += 1
        
        if quality_score:
            # Update running average
            current_avg = stats["avg_quality"]
            total_uses = stats["total_uses"]
            stats["avg_quality"] = (current_avg * (total_uses - 1) + quality_score) / total_uses

# ============================================================================
# EXISTING FUNCTIONS (Enhanced with agentic features)
# ============================================================================

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
    citations += f"*Based on analysis of {len(sources)} sources:*\n\n"
    
    # Separate papers and articles
    papers = []
    articles = []
    
    for source in sources:
        if "authors" in source:  # arXiv paper
            papers.append(source)
        elif "author" in source:  # AI Tech article
            articles.append(source)
    
    # Format arXiv papers
    if papers:
        citations += "### 📄 Academic Research Papers\n\n"
        for i, source in enumerate(papers, 1):
            citations += f"**{i}.** {source['title']}\n"
            citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
            citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
    
    # Format AI Tech articles
    if articles:
        citations += "### 🚀 AI Tech Articles\n\n"
        for i, source in enumerate(articles, 1):
            citations += f"**{i}.** {source['title']}\n"
            citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
            citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
    
    citations += "---\n*These sources were retrieved using semantic search through multiple databases.*"
    return citations

def summarize_papers(query: str) -> str:
    """
    Get a comprehensive summary of papers related to a topic.
    """
    try:
        # Get papers using intelligent search
        search_result = intelligent_search(query)
        if search_result["total_count"] == 0:
            return search_result["content"]
        
        # Ask the LLM to summarize
        summary_prompt = f"""
        Based on the following research content, provide a comprehensive summary of the current state of research on: {query}
        
        Content:
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
        
        # Track tool effectiveness
        tool_tracker.record_tool_usage("summarize_papers", query, True)
        
        return summary + sources_section
        
    except Exception as e:
        logger.error(f"Error in summarize_papers: {e}")
        tool_tracker.record_tool_usage("summarize_papers", query, False)
        return f"Error summarizing papers: {str(e)}"

def analyze_trends(topic: str) -> str:
    """
    Analyze trends in a specific research area.
    """
    try:
        # Search for recent content using intelligent search
        recent_query = f"latest developments {topic} 2024 2023"
        search_result = intelligent_search(recent_query)
        
        if search_result["total_count"] == 0:
            return f"No recent content found for {topic}"
        
        # Analyze trends
        trend_prompt = f"""
        Analyze the following recent content to identify trends in {topic}:
        
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
        
        # Track tool effectiveness
        tool_tracker.record_tool_usage("analyze_trends", topic, True)
        
        return analysis + sources_section
        
    except Exception as e:
        logger.error(f"Error in analyze_trends: {e}")
        tool_tracker.record_tool_usage("analyze_trends", topic, False)
        return f"Error analyzing trends: {str(e)}"

def intelligent_search(query: str) -> dict:
    """
    Intelligently search the most appropriate database(s) based on query analysis.
    
    Args:
        query (str): The search query
        
    Returns:
        dict: Dictionary containing search results from the selected database(s)
    """
    try:
        logger.info(f"Performing intelligent search for: {query}")
        
        # Use LLM to select the most appropriate database
        selected_database = select_database_for_query(query)
        
        combined_content = ""
        combined_sources = []
        
        # Search database1 (arXiv papers) if selected
        if selected_database == "database1" or selected_database == "both":
            try:
                logger.info(f"Searching database1 (arXiv) for: {query}")
                xq = embed.embed_query(query)
                
                out = database1_index.query(
                    vector=xq, 
                    top_k=3,
                    include_metadata=True,
                    include_values=False
                )
                
                logger.info(f"Database1 query returned {len(out.get('matches', []))} matches")
                
                if out["matches"]:
                    db1_content = "## 📚 Academic Research Papers\n\n"
                    db1_sources = []
                    
                    for i, match in enumerate(out["matches"]):
                        score = match.get("score", 0)
                        logger.info(f"Database1 match {i+1}: score={score:.3f}")
                        
                        if "metadata" in match and "text" in match["metadata"]:
                            title = match["metadata"].get("title", "Unknown Title")
                            authors = match["metadata"].get("authors", "Unknown Authors")
                            date = match["metadata"].get("date", "Unknown Date")
                            paper_id = match["metadata"].get("paper_id", "Unknown ID")
                            url = match["metadata"].get("url", "")
                            
                            if score > 0.3:  # Threshold for relevance (lowered from 0.7 for testing)
                                logger.info(f"Database1 match {i+1} PASSED threshold: {title}")
                                # Don't include raw text in main response, only in sources
                                db1_content += f"[Score: {score:.2f}] {title} by {authors} ({date})\n---\n"
                                
                                source_info = {
                                    "title": title,
                                    "authors": authors,
                                    "date": date,
                                    "paper_id": paper_id,
                                    "url": url,
                                    "relevance_score": score,
                                    "excerpt": match["metadata"]["text"][:200] + "..." if len(match["metadata"]["text"]) > 200 else match["metadata"]["text"],
                                    "database": "arXiv Papers"
                                }
                                db1_sources.append(source_info)
                            else:
                                logger.info(f"Database1 match {i+1} FAILED threshold: {title}")
                    
                    if db1_sources:
                        combined_content += db1_content + "\n"
                        combined_sources.extend(db1_sources)
                        logger.info(f"Database1 added {len(db1_sources)} sources")
                    else:
                        logger.warning("Database1: No sources passed the 0.3 threshold")
                        
            except Exception as e:
                logger.error(f"Error searching database1: {e}")
        
        # Search database2 (AI Tech articles) if selected
        if selected_database == "database2" or selected_database == "both":
            try:
                logger.info(f"Searching database2 (AI Tech) for: {query}")
                xq = embed.embed_query(query)
                
                out = database2_index.query(
                    vector=xq, 
                    top_k=3,
                    include_metadata=True,
                    include_values=False
                )
                
                logger.info(f"Database2 query returned {len(out.get('matches', []))} matches")
                
                if out["matches"]:
                    db2_content = "## 🚀 AI Tech Articles\n\n"
                    db2_sources = []
                    
                    for i, match in enumerate(out["matches"]):
                        score = match.get("score", 0)
                        logger.info(f"Database2 match {i+1}: score={score:.3f}")
                        
                        if "metadata" in match and "text" in match["metadata"]:
                            title = match["metadata"].get("title", "Unknown Title")
                            author = match["metadata"].get("author", "Unknown Author")
                            date = match["metadata"].get("date", "Unknown Date")
                            article_id = match["metadata"].get("article_id", "Unknown ID")
                            url = match["metadata"].get("url", "")
                            source = match["metadata"].get("source", "Unknown Source")
                            
                            if score > 0.3:  # Threshold for relevance (lowered from 0.7 for testing)
                                logger.info(f"Database2 match {i+1} PASSED threshold: {title}")
                                # Don't include raw text in main response, only in sources
                                db2_content += f"[Score: {score:.2f}] {title} by {author} ({source}, {date})\n---\n"
                                
                                source_info = {
                                    "title": title,
                                    "author": author,
                                    "date": date,
                                    "article_id": article_id,
                                    "url": url,
                                    "source": source,
                                    "relevance_score": score,
                                    "excerpt": match["metadata"]["text"][:200] + "..." if len(match["metadata"]["text"]) > 200 else match["metadata"]["text"],
                                    "database": "AI Tech Articles"
                                }
                                db2_sources.append(source_info)
                            else:
                                logger.info(f"Database2 match {i+1} FAILED threshold: {title}")
                    
                    if db2_sources:
                        combined_content += db2_content + "\n"
                        combined_sources.extend(db2_sources)
                        logger.info(f"Database2 added {len(db2_sources)} sources")
                    else:
                        logger.warning("Database2: No sources passed the 0.3 threshold")
                        
            except Exception as e:
                logger.error(f"Error searching database2: {e}")
        
        if not combined_content:
            return {
                "content": f"No relevant content found in the selected database(s) for this query. (Selected: {selected_database})",
                "sources": [],
                "total_count": 0,
                "selected_database": selected_database
            }
        
        # Add database selection info to the response
        db_info = f"\n\n*🔍 **Database Selection**: The AI selected '{selected_database}' for this query based on content analysis.*\n\n"
        combined_content = db_info + combined_content
        
        # Track tool effectiveness
        success = len(combined_sources) > 0
        tool_tracker.record_tool_usage("intelligent_search", query, success)
        
        return {
            "content": combined_content,
            "sources": combined_sources,
            "total_count": len(combined_sources),
            "selected_database": selected_database
        }
        
    except Exception as e:
        logger.error(f"Error in intelligent_search: {e}")
        tool_tracker.record_tool_usage("intelligent_search", query, False)
        return {
            "content": f"Error performing intelligent search: {str(e)}",
            "sources": [],
            "total_count": 0,
            "selected_database": "both"
        }

# Register enhanced tools for the agent
tools = [
    Tool.from_function(
        func=lambda query: intelligent_search(query)["content"],  # Extract content for tool
        name="intelligent_search",
        description="Use this tool to intelligently search the most appropriate database(s) for a query. "
                   "The AI analyzes the query content and automatically selects whether to search "
                   "academic papers (arXiv), AI Tech articles (industry), or both based on relevance. "
                   "This provides optimal results by matching query intent with the right knowledge base."
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

# Global chat history and cache with better memory management
chat_history = []
response_cache = {}
MAX_HISTORY_EXCHANGES = 6  # 3 exchanges (6 messages: 3 human + 3 assistant)
MAX_CACHE_SIZE = 100
CACHE_CLEANUP_SIZE = 20

# Query enhancement and classification system
QUERY_ENHANCEMENTS = {
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
    
    # Check for comparative queries
    if any(word in query_lower for word in ["compare", "difference", "vs", "versus", "versus", "contrast", "similarity"]):
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
    Enhance query based on its classification for better results.
    
    Args:
        query (str): Original query
        query_type (str): Query type classification
        
    Returns:
        str: Enhanced query
    """
    if not query_type:
        query_type = classify_query_type(query)
    
    enhancement = QUERY_ENHANCEMENTS.get(query_type, QUERY_ENHANCEMENTS["general"])
    enhanced_query = f"{query} {enhancement}"
    
    logger.info(f"Enhanced query: {enhanced_query}")
    return enhanced_query

def generate_narrative_response(query: str, search_result: dict) -> str:
    """
    Generate a narrative-style response with paragraphs first, then bullet points.
    
    Args:
        query (str): The user's question
        search_result (dict): Search results from intelligent_search
        
    Returns:
        str: Formatted narrative response
    """
    try:
        # Create a prompt for narrative response
        narrative_prompt = f"""
        Based on the following research content, provide a comprehensive response to: {query}
        
        Research Content:
        {search_result["content"]}
        
        Please structure your response as follows:
        
        1. **Narrative Section**: Write 2-3 paragraphs that provide a comprehensive overview of the topic, explaining key concepts, developments, and implications in a flowing narrative style.
        
        2. **Key Points Section**: After the narrative, provide key points in bullet format covering:
           - Current technological advances
           - Key research objectives  
           - Technological challenges
           - Promising demonstrations
           - Future potential
        
        Make the narrative engaging and informative, connecting different aspects of the research into a coherent story.
        """
        
        response = llm.invoke(narrative_prompt)
        narrative_response = str(response.content)
        
        # Add sources section
        sources_section = format_sources(search_result["sources"])
        
        return narrative_response + sources_section
        
    except Exception as e:
        logger.error(f"Error generating narrative response: {e}")
        # Fallback to simple response
        return f"Based on the research content: {search_result['content']}" + format_sources(search_result["sources"])

def chat(query: str) -> str:
    """
    Main chat function - simplified processing for all queries.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The AI assistant's response with sources
    """
    try:
        # Check cache for similar queries
        cache_key = query.lower().strip()
        if cache_key in response_cache:
            logger.info("Returning cached response")
            return response_cache[cache_key]
        
        # Enhance the query for better results
        query_type = classify_query_type(query)
        enhanced_query = enhance_query(query, query_type)
        
        # Add a small delay to prevent rate limiting
        time.sleep(1)
        
        # Get search results directly
        search_result = intelligent_search(enhanced_query)
        
        if search_result["total_count"] == 0:
            return f"I couldn't find relevant information for your query: {query}"
        
        # Generate narrative response
        narrative_response = generate_narrative_response(query, search_result)
        
        # Add agentic features
        enhanced_response = _add_agentic_features(narrative_response, query)
        
        # Cache and update history
        _update_cache_and_history(query, enhanced_response)
        
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        error_message = f"I apologize, but I encountered an error while processing your query: {str(e)}"
        return error_message

def _extract_sources(out: dict, query: str) -> str:
    """Extract sources from agent execution results."""
    sources_section = ""
    try:
        if "intermediate_steps" in out:
            for step in out["intermediate_steps"]:
                if step[0].tool == "intelligent_search":
                    search_result = intelligent_search(step[0].tool_input)
                    if search_result["sources"]:
                        sources_section = format_sources(search_result["sources"])
                        break
        
        # Fallback to direct search
        if not sources_section:
            direct_search = intelligent_search(query)
            if direct_search["sources"]:
                sources_section = format_sources(direct_search["sources"])
                
    except Exception as e:
        logger.warning(f"Could not extract sources: {e}")
    
    return sources_section

def _add_agentic_features(response: str, query: str) -> str:
    """Add agentic features to the response."""
    enhanced_response = response
    
    # Add follow-up questions
    follow_up_questions = proactive_agent.generate_follow_up_questions(query, response)
    if follow_up_questions:
        questions_section = "\n\n**🎯 Suggested Follow-up Questions:**\n"
        for question in follow_up_questions[:3]:
            questions_section += f"• {question}\n"
        enhanced_response += questions_section
    
    # Add research directions
    research_directions = proactive_agent.suggest_research_directions(query, response)
    if research_directions:
        directions_section = "\n\n**🚀 Suggested Research Directions:**\n"
        for direction in research_directions:
            directions_section += f"• {direction}\n"
        enhanced_response += directions_section
    
    return enhanced_response

def _update_cache_and_history(query: str, response: str):
    """Update cache and chat history."""
    global chat_history, response_cache
    
    # Cache the response
    cache_key = query.lower().strip()
    response_cache[cache_key] = response
    
    # Cache management
    if len(response_cache) > MAX_CACHE_SIZE:
        oldest_keys = list(response_cache.keys())[:CACHE_CLEANUP_SIZE]
        for key in oldest_keys:
            del response_cache[key]
    
    # Update chat history
    exchange = {
        "human": query,
        "assistant": response,
        "timestamp": datetime.now(),
        "query_type": classify_query_type(query)
    }
    chat_history.append(exchange)
    
    # Limit history size
    if len(chat_history) > MAX_HISTORY_EXCHANGES:
        chat_history = chat_history[-MAX_HISTORY_EXCHANGES:]

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
        "avg_exchange_length": calculate_avg_exchange_length(),
        "tool_effectiveness": tool_tracker.tool_stats,
        "agentic_features_used": len([h for h in chat_history if "🤖 Agent Confidence" in h.get("assistant", "")])
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
    search_result = intelligent_search(test_query)
    logger.info(f"Search result: {search_result['total_count']} sources found")
    
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

def get_agentic_insights():
    """Get insights about agentic behavior and performance."""
    return {
        "tool_effectiveness": tool_tracker.tool_stats,
        "proactive_suggestions": len([h for h in chat_history if "Suggested Follow-up Questions" in h.get("assistant", "")])
    }

# Initialize simplified agentic components
proactive_agent = ProactiveAgent(llm)
tool_tracker = ToolEffectivenessTracker()
