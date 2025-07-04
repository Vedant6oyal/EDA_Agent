# insight_tool.py

from langchain.tools import Tool
import pandas as pd
from data_loader import load_snapshots
import os
from dotenv import load_dotenv
import google.generativeai as genai
import io
import base64
from PIL import Image as PILImage

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Persistent context of last visualization (shared across tools)
last_chart_summary = ""

def analyze_chart_with_gemini(query, df, figure=None):
    """Analyze chart using Gemini 2.0 Flash vision model"""
    
    # Get the last generated figure if not provided
    if figure is None:
        try:
            from tools.plot_tool import get_last_figure
            figure = get_last_figure()
        except ImportError:
            figure = None
    
    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    if figure is not None:
        # Convert matplotlib figure to PIL Image
        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        pil_image = PILImage.open(buffer)
        
        # Create prompt for vision analysis
        vision_prompt = f"""
You are a senior data analyst. Analyze this visualization and provide detailed insights.

User Query: "{query}"

Dataset Context:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Sample data: {df.head(2).to_string()}

Please provide:
1. **Chart Description**: What type of visualization is this and what does it show?
2. **Key Patterns**: What are the main trends, correlations, or patterns visible?
3. **Statistical Insights**: What statistical relationships can you observe?
4. **Business Implications**: What do these patterns mean for decision-making?
5. **Actionable Recommendations**: What actions should be taken based on these insights?

Be specific about what you see in the chart - mention actual values, trends, outliers, and relationships.
"""
        
        try:
            # Send image and prompt to Gemini
            response = model.generate_content([vision_prompt, pil_image])
            return response.text
            
        except Exception as e:
            return f"❌ Error analyzing chart with Gemini: {str(e)}"
    
    else:
        # Fallback to text-only analysis if no figure available
        text_prompt = f"""
You are a senior data analyst. A user asked: "{query}"

Dataset Information:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Sample data: {df.head(3).to_string()}

Provide comprehensive insights about this dataset focusing on:
- Key patterns and relationships
- Statistical observations
- Business implications
- Actionable recommendations

Be specific and detailed in your analysis.
"""
        
        try:
            response = model.generate_content(text_prompt)
            return response.text
            
        except Exception as e:
            return f"❌ Error generating insights with Gemini: {str(e)}"

def set_chart_summary(text):
    global last_chart_summary
    last_chart_summary = text

# Create the LangChain tool
insight_tool = Tool.from_function(
    name="GeminiVisionInsights",
    func=lambda q: analyze_chart_with_gemini(q, df=load_snapshots()),
    description=(
        "Use this tool to get detailed visual analysis and insights from charts and data. "
        "Can analyze actual visualizations using Gemini's vision capabilities. "
        "Provides comprehensive insights about patterns, trends, and business implications."
    )
)
