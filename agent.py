from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from memory import get_memory
import os
from dotenv import load_dotenv

from tools.plot_tool import dynamic_python_tool
from data_loader import load_snapshots

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.2,
    max_tokens=1024,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

tools = [dynamic_python_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=get_memory(),
    handle_parsing_errors=True
)
