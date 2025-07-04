from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from memory import get_memory
import os
from dotenv import load_dotenv

from tools.plot_tool import dynamic_python_tool
from tools.insight_tool import insight_tool
from data_loader import load_snapshots

load_dotenv()

# Load LLM (Claude)
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.3,
    max_tokens=4096,  # Increased for longer responses
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Add memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Assemble tools
tools = [
    dynamic_python_tool,  # For visualizations
    insight_tool          # For Gemini vision insights
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)
