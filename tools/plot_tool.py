from langchain.tools import Tool
from anthropic import Anthropic
import pandas as pd
from data_loader import load_snapshots
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")    
client = Anthropic(api_key=api_key)

# Global variable to store the last generated figure
_last_figure = None
_last_query = ""

def generate_and_run_code(query, df):
    global _last_figure, _last_query
    code = ""  # ensure code is defined even if prompt fails

    prompt = f"""
You are a Python data analyst. A user asked: "{query}"

Here is the first 3 rows of the DataFrame `df`:
{df.head(3).to_string()}

Write Python code (only code, no explanation) that:
- Uses pandas, seaborn, or matplotlib
- Assumes df is already loaded
- Always begins with: fig = plt.figure(figsize=(10,6))
- Draws plots into that figure
- Ends with: return fig (do NOT use plt.show())
- Handles missing values and categorical axes if needed
- Do NOT include any explanation or markdown formatting. Only return raw, executable Python code. No text before or after.  
"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            max_tokens=3096,
            messages=[{"role": "user", "content": prompt}]
        )

        code = response.content[0].text.strip()

        # Clean backticks if present
        if code.startswith("```"):
            code = code.replace("```python", "").replace("```", "").strip()

        exec_globals = {
            "df": df,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "np": __import__("numpy"),
        }

        # Wrap the code in a function for better scoping
        wrapped_code = "def execute_code():\n    " + "\n    ".join(code.splitlines()) + "\n    return fig"
        local_vars = {}
        exec(wrapped_code, exec_globals, local_vars)
        fig = local_vars["execute_code"]()
        
        # Store the figure globally so we can access it later
        _last_figure = fig
        _last_query = query
        
        # Set context for insight tool
        try:
            from tools.insight_tool import set_chart_summary
            set_chart_summary(f"Created visualization for query: '{query}'. The chart shows data analysis results.")
        except ImportError:
            pass  # insight_tool might not be available
        
        # Return a success message that indicates a figure was created
        return f"✅ Successfully generated visualization with dimensions {fig.get_size_inches()[0]*fig.dpi:.0f}x{fig.get_size_inches()[1]*fig.dpi:.0f}"

    except Exception as e:
        return f"❌ Error generating or executing code: {str(e)}\n\n🧠 Generated code:\n{code}"

def get_last_figure():
    """Get the last generated figure"""
    global _last_figure
    return _last_figure

def get_last_query():
    """Get the last visualization query"""
    global _last_query
    return _last_query

# Define the LangChain Tool
dynamic_python_tool = Tool.from_function(
    name="DynamicPythonChart",
    func=lambda q: generate_and_run_code(q, df=load_snapshots()),
    description=(
        "Use this tool when the user asks for any kind of data visualization or analysis using the dataset. "
        "Generates matplotlib/seaborn charts dynamically based on natural language input."
    )
)
