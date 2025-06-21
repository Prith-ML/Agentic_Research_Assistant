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
    Search through arXiv papers using vector similarity.
    
    Args:
        query (str): The search query
        
    Returns:
        str: Relevant paper excerpts joined with separators
    """
    try:
        logger.info(f"Searching for: {query}")
        xq = embed.embed_query(query)
        out = index.query(vector=xq, top_k=5, include_metadata=True)
        
        if not out["matches"]:
            return "No relevant papers found for this query."
        
        results = []
        for match in out["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                results.append(match["metadata"]["text"])
        
        return "\n---\n".join(results)
        
    except Exception as e:
        logger.error(f"Error in arxiv_search: {e}")
        return f"Error searching papers: {str(e)}"

# Register tools for the agent
tools = [
    Tool.from_function(
        func=arxiv_search,
        name="arxiv_search",
        description="Use this tool to answer questions about AI, ML, or arXiv papers. "
                   "This tool searches through a curated dataset of scientific papers "
                   "and returns relevant excerpts to help answer research questions."
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

# Global chat history
chat_history = ""

def chat(query: str) -> str:
    """
    Main chat function that processes user queries and returns responses.
    
    Args:
        query (str): The user's question
        
    Returns:
        str: The AI assistant's response
    """
    global chat_history
    
    try:
        logger.info(f"Processing query: {query}")
        
        # Add a small delay to prevent rate limiting
        time.sleep(1)
        
        # Execute the agent
        out = agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        answer = out.get("output", "I apologize, but I couldn't generate a response for your query.")
        
        # Update chat history
        chat_history += f"\nHuman: {query}\nAssistant: {answer}"
        
        logger.info("Query processed successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        error_message = f"I apologize, but I encountered an error while processing your query: {str(e)}"
        return error_message


# In[ ]:
