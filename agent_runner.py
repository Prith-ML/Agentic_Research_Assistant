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
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
    # Initialize both databases
    database1_index = pc.Index("database1")  # arXiv research papers
    database2_index = pc.Index("database2")  # AI Tech articles
    logger.info("Pinecone databases initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

# ============================================================================
# AGENTIC AI SYSTEM - NEW COMPONENTS
# ============================================================================

@dataclass
class SubGoal:
    """Represents a sub-goal in a research plan."""
    description: str
    priority: float
    estimated_time: int  # minutes
    required_tools: List[str]
    status: str = "pending"  # pending, in_progress, completed, failed

@dataclass
class ResearchPlan:
    """Represents a complete research plan."""
    main_goal: str
    sub_goals: List[SubGoal]
    total_estimated_time: int
    current_step: int = 0
    progress: float = 0.0

@dataclass
class Gap:
    """Represents an information gap in a response."""
    topic: str
    importance: float
    suggested_sources: List[str]
    description: str

@dataclass
class Reflection:
    """Represents self-reflection on response quality."""
    quality_score: float
    gaps: List[Gap]
    improvement_suggestions: List[str]
    confidence_level: float

class GoalDecomposer:
    """Decomposes complex research goals into manageable sub-tasks."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, user_goal: str) -> List[SubGoal]:
        """Break down a complex goal into specific sub-goals."""
        prompt = f"""
        Break down this research goal into 3-5 specific sub-tasks:
        
        GOAL: {user_goal}
        
        For each sub-task, provide:
        1. A clear description of what needs to be accomplished
        2. Priority level (0.1 to 1.0, where 1.0 is highest)
        3. Estimated time in minutes
        4. Required tools (arxiv_search, summarize_papers, analyze_trends)
        
        Return as JSON format:
        {{
            "sub_goals": [
                {{
                    "description": "string",
                    "priority": float,
                    "estimated_time": int,
                    "required_tools": ["string"]
                }}
            ]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            plan_data = json.loads(response.content)
            
            sub_goals = []
            for goal_data in plan_data["sub_goals"]:
                sub_goal = SubGoal(
                    description=goal_data["description"],
                    priority=goal_data["priority"],
                    estimated_time=goal_data["estimated_time"],
                    required_tools=goal_data["required_tools"]
                )
                sub_goals.append(sub_goal)
            
            logger.info(f"Decomposed goal into {len(sub_goals)} sub-goals")
            return sub_goals
            
        except Exception as e:
            logger.error(f"Error decomposing goal: {e}")
            # Fallback to simple decomposition
            return self._simple_decompose(user_goal)
    
    def _simple_decompose(self, user_goal: str) -> List[SubGoal]:
        """Simple fallback goal decomposition."""
        return [
            SubGoal(
                description=f"Research {user_goal}",
                priority=1.0,
                estimated_time=10,
                required_tools=["arxiv_search"]
            )
        ]

class SelfReflectionAgent:
    """Evaluates response quality and identifies improvement areas."""
    
    def __init__(self, llm):
        self.llm = llm
        self.quality_history = []
    
    def evaluate_response(self, query: str, response: str, sources: List[Dict] = None) -> Reflection:
        """Comprehensive evaluation of response quality."""
        
        # Evaluate overall quality
        quality_score = self._assess_quality(query, response)
        
        # Identify information gaps
        gaps = self._identify_gaps(query, response)
        
        # Generate improvement suggestions
        suggestions = self._generate_suggestions(query, response, gaps)
        
        # Calculate confidence level
        confidence = self._calculate_confidence(response, sources or [])
        
        reflection = Reflection(
            quality_score=quality_score,
            gaps=gaps,
            improvement_suggestions=suggestions,
            confidence_level=confidence
        )
        
        # Store for learning
        self.quality_history.append({
            "query": query,
            "response": response,
            "reflection": reflection,
            "timestamp": datetime.now()
        })
        
        logger.info(f"Response evaluation: Quality={quality_score:.2f}, Confidence={confidence:.2f}")
        return reflection
    
    def _assess_quality(self, query: str, response: str) -> float:
        """Assess response quality on multiple dimensions."""
        prompt = f"""
        Rate the quality of this response (0.0 to 1.0) for the query: "{query}"
        
        Response: {response}
        
        Consider these dimensions:
        1. Relevance to the query (0-1)
        2. Completeness of information (0-1)
        3. Clarity and organization (0-1)
        4. Depth of analysis (0-1)
        
        Return only a single number between 0.0 and 1.0 representing overall quality.
        """
        
        try:
            response_text = self.llm.invoke(prompt).content.strip()
            return float(response_text)
        except:
            return 0.7  # Default fallback
    
    def _identify_gaps(self, query: str, response: str) -> List[Gap]:
        """Identify information gaps in the response."""
        prompt = f"""
        For the query: "{query}"
        And response: "{response}"
        
        Identify what information might be missing or could be expanded.
        Focus on:
        1. Topics that were mentioned but not fully explained
        2. Related areas that could provide context
        3. Recent developments that might be relevant
        4. Practical applications or implications
        
        Return as JSON:
        {{
            "gaps": [
                {{
                    "topic": "string",
                    "importance": float,
                    "description": "string",
                    "suggested_sources": ["string"]
                }}
            ]
        }}
        """
        
        try:
            response_text = self.llm.invoke(prompt).content
            gaps_data = json.loads(response_text)
            
            gaps = []
            for gap_data in gaps_data["gaps"]:
                gap = Gap(
                    topic=gap_data["topic"],
                    importance=gap_data["importance"],
                    description=gap_data["description"],
                    suggested_sources=gap_data["suggested_sources"]
                )
                gaps.append(gap)
            
            return gaps
        except:
            return []
    
    def _generate_suggestions(self, query: str, response: str, gaps: List[Gap]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Add gap-based suggestions
        for gap in gaps[:3]:  # Top 3 gaps
            if gap.importance > 0.7:
                suggestions.append(f"Explore {gap.topic} in more detail")
        
        # Add general suggestions
        if len(response) < 500:
            suggestions.append("Provide more detailed explanations")
        
        if not any(word in response.lower() for word in ["recent", "latest", "2024", "2023"]):
            suggestions.append("Include more recent developments")
        
        return suggestions
    
    def _calculate_confidence(self, response: str, sources: List[Dict] = None) -> float:
        """Calculate confidence level based on response and sources."""
        confidence = 0.7  # Base confidence
        
        # Boost confidence based on source quality
        if sources:
            avg_source_score = sum(s.get("relevance_score", 0.5) for s in sources) / len(sources)
            confidence += avg_source_score * 0.2
        
        # Boost confidence based on response length and structure
        if len(response) > 1000:
            confidence += 0.1
        
        if "##" in response:  # Has sections
            confidence += 0.05
        
        return min(confidence, 1.0)

class ProactiveAgent:
    """Generates proactive suggestions and follow-up questions."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_follow_up_questions(self, query: str, response: str, gaps: List[Gap]) -> List[str]:
        """Generate intelligent follow-up questions."""
        questions = []
        
        # Add gap-based questions
        for gap in gaps[:3]:
            if gap.importance > 0.6:
                question = f"Would you like me to explore {gap.topic} in more detail?"
                questions.append(question)
        
        # Generate contextual follow-ups
        prompt = f"""
        Based on the query: "{query}"
        And the response provided, suggest 2-3 follow-up questions that would be helpful.
        
        Focus on:
        - Areas that could be explored further
        - Related topics of interest
        - Practical applications or implications
        - Recent developments or trends
        
        Return as a simple list, one question per line.
        """
        
        try:
            response_text = self.llm.invoke(prompt).content
            contextual_questions = [q.strip() for q in response_text.split('\n') if q.strip()]
            questions.extend(contextual_questions[:3])
        except:
            pass
        
        return questions[:5]  # Limit to 5 questions
    
    def suggest_research_directions(self, query: str, response: str) -> List[str]:
        """Suggest new research directions based on current findings."""
        prompt = f"""
        Based on the research query: "{query}"
        And the findings: "{response}"
        
        Suggest 2-3 new research directions or areas to explore that would be valuable.
        
        Focus on:
        - Emerging trends or gaps
        - Related fields or applications
        - Future research opportunities
        
        Return as a simple list, one direction per line.
        """
        
        try:
            response_text = self.llm.invoke(prompt).content
            directions = [d.strip() for d in response_text.split('\n') if d.strip()]
            return directions[:3]
        except:
            return []

class ToolEffectivenessTracker:
    """Tracks and optimizes tool effectiveness."""
    
    def __init__(self):
        self.tool_stats = {}
        self.query_tool_mapping = {}
    
    def record_tool_usage(self, tool_name: str, query: str, success: bool, quality_score: float = None):
        """Record tool usage and effectiveness."""
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = {
                "total_uses": 0,
                "successful_uses": 0,
                "avg_quality": 0.0,
                "query_types": {}
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
        
        # Track query type effectiveness
        query_type = self._classify_query_type(query)
        if query_type not in stats["query_types"]:
            stats["query_types"][query_type] = {"uses": 0, "successes": 0}
        
        stats["query_types"][query_type]["uses"] += 1
        if success:
            stats["query_types"][query_type]["successes"] += 1
    
    def get_best_tool_for_query(self, query: str) -> str:
        """Recommend the best tool for a given query."""
        query_type = self._classify_query_type(query)
        
        best_tool = "arxiv_search"  # Default
        best_score = 0.0
        
        for tool_name, stats in self.tool_stats.items():
            if query_type in stats["query_types"]:
                type_stats = stats["query_types"][query_type]
                if type_stats["uses"] > 0:
                    success_rate = type_stats["successes"] / type_stats["uses"]
                    if success_rate > best_score:
                        best_score = success_rate
                        best_tool = tool_name
        
        return best_tool
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for tool selection."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "difference", "vs", "versus"]):
            return "comparative"
        elif any(word in query_lower for word in ["trend", "latest", "recent", "development"]):
            return "trend"
        elif any(word in query_lower for word in ["summary", "review", "overview"]):
            return "summary"
        elif any(word in query_lower for word in ["how", "implement", "code"]):
            return "implementation"
        else:
            return "general"

class ResearchPlanner:
    """Creates and manages research plans."""
    
    def __init__(self, goal_decomposer: GoalDecomposer):
        self.goal_decomposer = goal_decomposer
    
    def create_research_plan(self, goal: str) -> ResearchPlan:
        """Create a comprehensive research plan."""
        sub_goals = self.goal_decomposer.decompose(goal)
        
        total_time = sum(goal.estimated_time for goal in sub_goals)
        
        plan = ResearchPlan(
            main_goal=goal,
            sub_goals=sub_goals,
            total_estimated_time=total_time
        )
        
        logger.info(f"Created research plan with {len(sub_goals)} sub-goals, estimated {total_time} minutes")
        return plan
    
    def execute_plan(self, plan: ResearchPlan) -> List[Dict]:
        """Execute a research plan step by step."""
        results = []
        
        for i, sub_goal in enumerate(plan.sub_goals):
            logger.info(f"Executing sub-goal {i+1}/{len(plan.sub_goals)}: {sub_goal.description}")
            
            # Execute sub-goal
            result = self._execute_sub_goal(sub_goal)
            results.append(result)
            
            # Update plan progress
            plan.current_step = i + 1
            plan.progress = (i + 1) / len(plan.sub_goals)
            sub_goal.status = "completed"
        
        return results
    
    def _execute_sub_goal(self, sub_goal: SubGoal) -> Dict:
        """Execute a single sub-goal."""
        # Use the best tool for this sub-goal
        tool_name = sub_goal.required_tools[0] if sub_goal.required_tools else "arxiv_search"
        
        if tool_name == "arxiv_search":
            result = arxiv_search(sub_goal.description)
        elif tool_name == "summarize_papers":
            result = summarize_papers(sub_goal.description)
        elif tool_name == "analyze_trends":
            result = analyze_trends(sub_goal.description)
        else:
            result = arxiv_search(sub_goal.description)
        
        return {
            "sub_goal": sub_goal.description,
            "tool_used": tool_name,
            "result": result,
            "status": "completed"
        }

# Initialize agentic components
goal_decomposer = GoalDecomposer(llm)
self_reflection_agent = SelfReflectionAgent(llm)
proactive_agent = ProactiveAgent(llm)
tool_tracker = ToolEffectivenessTracker()
research_planner = ResearchPlanner(goal_decomposer)

# ============================================================================
# EXISTING FUNCTIONS (Enhanced with agentic features)
# ============================================================================

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
        out = database1_index.query(
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
        
        # Track tool effectiveness
        success = len(sources) > 0
        tool_tracker.record_tool_usage("arxiv_search", query, success)
        
        return {
            "content": "\n---\n".join(results[:5]),  # Return top 5 most relevant
            "sources": sources[:5],  # Top 5 sources
            "paper_count": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Error in arxiv_search: {e}")
        tool_tracker.record_tool_usage("arxiv_search", query, False)
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
            citations += f"   - **Authors:** {source['authors']}\n"
            citations += f"   - **Date:** {source['date']}\n"
            citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
            if source.get('url'):
                citations += f"   - **Link:** [View Paper]({source['url']})\n"
            citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
    
    # Format AI Tech articles
    if articles:
        citations += "### 🚀 AI Tech Articles\n\n"
        for i, source in enumerate(articles, 1):
            citations += f"**{i}.** {source['title']}\n"
            citations += f"   - **Author:** {source['author']}\n"
            citations += f"   - **Source:** {source['source']}\n"
            citations += f"   - **Date:** {source['date']}\n"
            citations += f"   - **Relevance Score:** {source['relevance_score']:.2f}\n"
            if source.get('url'):
                citations += f"   - **Link:** [View Article]({source['url']})\n"
            citations += f"   - **Excerpt:** {source['excerpt']}\n\n"
    
    citations += "---\n*These sources were retrieved using semantic search through multiple databases.*"
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
        
        # Track tool effectiveness
        tool_tracker.record_tool_usage("analyze_trends", topic, True)
        
        return analysis + sources_section
        
    except Exception as e:
        logger.error(f"Error in analyze_trends: {e}")
        tool_tracker.record_tool_usage("analyze_trends", topic, False)
        return f"Error analyzing trends: {str(e)}"

def ai_tech_search(query: str) -> dict:
    """
    Search through AI Tech articles using vector similarity.
    
    Args:
        query (str): The search query
        
    Returns:
        dict: Dictionary containing search results and source information
    """
    try:
        logger.info(f"Searching AI Tech articles for: {query}")
        xq = embed.embed_query(query)
        
        # Enhanced query with better parameters
        out = database2_index.query(
            vector=xq, 
            top_k=8,  # Increased for better coverage
            include_metadata=True,
            include_values=False
        )
        
        if not out["matches"]:
            return {
                "content": "No relevant AI Tech articles found for this query.",
                "sources": [],
                "article_count": 0
            }
        
        # Enhanced result processing with ranking and source tracking
        results = []
        sources = []
        
        for match in out["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                score = match.get("score", 0)
                title = match["metadata"].get("title", "Unknown Title")
                author = match["metadata"].get("author", "Unknown Author")
                date = match["metadata"].get("date", "Unknown Date")
                article_id = match["metadata"].get("article_id", "Unknown ID")
                url = match["metadata"].get("url", "")
                source = match["metadata"].get("source", "Unknown Source")
                
                # Only include high-quality matches
                if score > 0.7:  # Threshold for relevance
                    result_text = f"[Score: {score:.2f}] {title} by {author} ({source}, {date})\n{match['metadata']['text']}"
                    results.append(result_text)
                    
                    # Add source information
                    source_info = {
                        "title": title,
                        "author": author,
                        "date": date,
                        "article_id": article_id,
                        "url": url,
                        "source": source,
                        "relevance_score": score,
                        "excerpt": match["metadata"]["text"][:200] + "..." if len(match["metadata"]["text"]) > 200 else match["metadata"]["text"]
                    }
                    sources.append(source_info)
        
        if not results:
            return {
                "content": "Found articles but none met the relevance threshold. Try rephrasing your question.",
                "sources": [],
                "article_count": 0
            }
        
        # Track tool effectiveness
        success = len(sources) > 0
        tool_tracker.record_tool_usage("ai_tech_search", query, success)
        
        return {
            "content": "\n---\n".join(results[:5]),  # Return top 5 most relevant
            "sources": sources[:5],  # Top 5 sources
            "article_count": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Error in ai_tech_search: {e}")
        tool_tracker.record_tool_usage("ai_tech_search", query, False)
        return {
            "content": f"Error searching AI Tech articles: {str(e)}",
            "sources": [],
            "article_count": 0
        }

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
        
        # Search based on LLM's decision
        if selected_database == "database1" or selected_database == "both":
            arxiv_results = arxiv_search(query)
            if arxiv_results["paper_count"] > 0:
                combined_content += "## 📚 Academic Research Papers\n\n"
                combined_content += arxiv_results["content"] + "\n\n"
                combined_sources.extend(arxiv_results["sources"])
        
        if selected_database == "database2" or selected_database == "both":
            ai_tech_results = ai_tech_search(query)
            if ai_tech_results["article_count"] > 0:
                combined_content += "## 🚀 AI Tech Articles\n\n"
                combined_content += ai_tech_results["content"] + "\n\n"
                combined_sources.extend(ai_tech_results["sources"])
        
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
    ),
    Tool.from_function(
        func=ai_tech_search,
        name="ai_tech_search",
        description="Use this tool to search through AI Tech articles related to a specific topic. "
                   "This tool searches through a curated dataset of AI Tech articles "
                   "and returns relevant excerpts to help answer research questions."
    ),
    Tool.from_function(
        func=lambda query: intelligent_search(query)["content"],  # Extract content for tool
        name="intelligent_search",
        description="Use this tool to intelligently search the most appropriate database(s) for a query. "
                   "The AI analyzes the query content and automatically selects whether to search "
                   "academic papers (arXiv), AI Tech articles (industry), or both based on relevance. "
                   "This provides optimal results by matching query intent with the right knowledge base."
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

def is_complex_goal(query: str) -> bool:
    """
    Determine if a query represents a complex research goal that needs planning.
    
    Args:
        query (str): The user's query
        
    Returns:
        bool: True if this is a complex goal requiring planning
    """
    complex_keywords = [
        "comprehensive", "complete", "thorough", "research", "study", "analysis",
        "investigate", "explore", "examine", "understand", "learn about",
        "compare", "evaluate", "assess", "review", "survey"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in complex_keywords)

def agentic_chat(query: str) -> str:
    """
    Enhanced agentic chat function with goal-oriented planning, self-reflection, and proactive behavior.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The AI assistant's response with sources and agentic features
    """
    global chat_history, response_cache
    
    try:
        logger.info(f"Processing agentic query: {query}")
        
        # Check cache for similar queries
        cache_key = query.lower().strip()
        if cache_key in response_cache:
            logger.info("Returning cached response")
            return response_cache[cache_key]
        
        # Determine if this is a complex goal requiring planning
        if is_complex_goal(query):
            logger.info("Complex goal detected - using agentic planning")
            return _handle_complex_goal(query)
        else:
            logger.info("Simple query - using standard processing")
            return _handle_simple_query(query)
        
    except Exception as e:
        logger.error(f"Error in agentic_chat function: {e}")
        error_message = f"I apologize, but I encountered an error while processing your query: {str(e)}"
        return error_message

def _handle_complex_goal(query: str) -> str:
    """Handle complex research goals with planning and execution."""
    
    # Create research plan
    plan = research_planner.create_research_plan(query)
    
    # Execute the plan
    results = research_planner.execute_plan(plan)
    
    # Synthesize results
    synthesis_prompt = f"""
    Based on the research plan for: "{query}"
    
    Research Results:
    {json.dumps([r['result'] for r in results], indent=2)}
    
    Please provide a comprehensive response that:
    1. Addresses the original goal
    2. Synthesizes findings from all sub-goals
    3. Provides insights and conclusions
    4. Identifies key patterns and trends
    """
    
    response = llm.invoke(synthesis_prompt)
    answer = str(response.content)
    
    # Add sources from all results
    all_sources = []
    for result in results:
        if isinstance(result['result'], dict) and 'sources' in result['result']:
            all_sources.extend(result['result']['sources'])
    
    sources_section = format_sources(all_sources[:5])  # Top 5 sources
    
    # Self-evaluate the response
    reflection = self_reflection_agent.evaluate_response(query, answer, all_sources)
    
    # Add agentic features to response
    enhanced_response = _add_agentic_features(answer + sources_section, query, reflection)
    
    # Cache and update history
    _update_cache_and_history(query, enhanced_response)
    
    return enhanced_response

def _handle_simple_query(query: str) -> str:
    """Handle simple queries with standard processing."""
    
    # Enhance the query for better results
    query_type = classify_query_type(query)
    enhanced_query = enhance_query(query, query_type)
    
    # Add a small delay to prevent rate limiting
    time.sleep(1)
    
    # Extract context from recent conversation
    recent_exchanges = chat_history[-3:] if len(chat_history) >= 3 else chat_history
    recent_context = ""
    
    if recent_exchanges:
        context_parts = []
        for exchange in recent_exchanges:
            context_parts.append(f"Human: {exchange['human']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        recent_context = " ".join(context_parts)
    
    # Execute the agent with enhanced query and context
    out = agent_executor.invoke({
        "input": enhanced_query,
        "chat_history": recent_context
    })
    
    answer = out.get("output", "I apologize, but I couldn't generate a response for your query.")
    
    # Extract sources
    sources_section = _extract_sources(out, query)
    
    # Combine answer with sources
    full_response = answer + sources_section
    
    # Self-evaluate the response
    reflection = self_reflection_agent.evaluate_response(query, full_response)
    
    # Add agentic features
    enhanced_response = _add_agentic_features(full_response, query, reflection)
    
    # Cache and update history
    _update_cache_and_history(query, enhanced_response)
    
    return enhanced_response

def _extract_sources(out: dict, query: str) -> str:
    """Extract sources from agent execution results."""
    sources_section = ""
    try:
        if "intermediate_steps" in out:
            for step in out["intermediate_steps"]:
                if step[0].tool == "arxiv_search":
                    search_result = arxiv_search(step[0].tool_input)
                    if search_result["sources"]:
                        sources_section = format_sources(search_result["sources"])
                        break
        
        # Fallback to direct search
        if not sources_section:
            direct_search = arxiv_search(query)
            if direct_search["sources"]:
                sources_section = format_sources(direct_search["sources"])
                
    except Exception as e:
        logger.warning(f"Could not extract sources: {e}")
    
    return sources_section

def _add_agentic_features(response: str, query: str, reflection: Reflection) -> str:
    """Add agentic features to the response."""
    enhanced_response = response
    
    # Add confidence and quality indicators
    confidence_indicator = f"\n\n**🤖 Agent Confidence:** {reflection.confidence_level:.1%}"
    quality_indicator = f"**📊 Response Quality:** {reflection.quality_score:.1%}"
    enhanced_response += f"\n\n{confidence_indicator} | {quality_indicator}"
    
    # Add gap information if significant gaps exist
    if reflection.gaps and any(gap.importance > 0.7 for gap in reflection.gaps):
        gap_section = "\n\n**🔍 Information Gaps Identified:**\n"
        for gap in reflection.gaps[:2]:  # Top 2 gaps
            if gap.importance > 0.7:
                gap_section += f"• **{gap.topic}** (Importance: {gap.importance:.1%})\n"
        enhanced_response += gap_section
    
    # Add improvement suggestions
    if reflection.improvement_suggestions:
        suggestions_section = "\n\n**💡 Improvement Suggestions:**\n"
        for suggestion in reflection.improvement_suggestions[:2]:
            suggestions_section += f"• {suggestion}\n"
        enhanced_response += suggestions_section
    
    # Add follow-up questions
    follow_up_questions = proactive_agent.generate_follow_up_questions(query, response, reflection.gaps)
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

def chat(query: str) -> str:
    """
    Main chat function - now uses agentic processing.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The AI assistant's response with sources
    """
    return agentic_chat(query)

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
    search_result = arxiv_search(test_query)
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

def get_agentic_insights():
    """Get insights about agentic behavior and performance."""
    return {
        "tool_effectiveness": tool_tracker.tool_stats,
        "quality_trends": self_reflection_agent.quality_history[-10:] if self_reflection_agent.quality_history else [],
        "research_plans_created": len([h for h in chat_history if "Complex goal detected" in str(h)]),
        "proactive_suggestions": len([h for h in chat_history if "Suggested Follow-up Questions" in h.get("assistant", "")])
    }

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
        logger.error("No sources found in test search")
        return False
