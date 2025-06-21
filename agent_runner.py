#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import time
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor
from langchain.agents.xml.base import XMLAgentOutputParser
from langchain import hub
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_anthropic import ChatAnthropic
from pinecone import Pinecone


# In[3]:


CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


# In[5]:


assert CLAUDE_API_KEY is not None, "Claude API key missing."
assert PINECONE_API_KEY is not None, "Pinecone API key missing."


# In[7]:


llm = ChatAnthropic(
    model_name="claude-3-5-haiku-20241022",
    temperature=0.2,
    anthropic_api_key=CLAUDE_API_KEY
)


# In[9]:


embed = BedrockEmbeddings(
    model_id="cohere.embed-english-v3",
    region_name="us-east-1"
)


# In[11]:


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ragas")  # Replace with your index name


# In[13]:


# Define the search tool
def arxiv_search(query: str) -> str:
    """
    Use this tool when answering questions about AI, machine learning, data
    science, or other technical questions that may be answered using arXiv papers.
    """
    xq = embed.embed_query(query)
    out = index.query(vector=xq, top_k=5, include_metadata=True)
    return "\n---\n".join([x["metadata"]["text"] for x in out["matches"]])

# Register it for the agent
tools = [
    Tool.from_function(
        func=arxiv_search,
        name="arxiv_search",
        description="Use this tool to answer questions about AI, ML, or arXiv papers."
    )
]


# In[15]:


# Load XML agent prompt
prompt = hub.pull("hwchase17/xml-agent-convo")


# In[19]:


# XML formatting helpers
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += f"<tool>{action.tool}</tool><tool_input>{action.tool_input}</tool_input>"
        log += f"<observation>{observation}</observation>"
    return log

def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])


# In[21]:


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


# In[23]:


agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, verbose=True)

chat_history = ""

def chat(query: str) -> str:
    global chat_history
    time.sleep(1)

    out = agent_executor.invoke({
        "input": query,
        "chat_history": chat_history
    })

    answer = out.get("output", "No response.")
    chat_history += f"\nHuman: {query}\nAssistant: {answer}"
    return answer


# In[ ]:
