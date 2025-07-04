from typing import Dict, Any, List, Optional
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from config import Config

class PromptTemplates:
    """
    Centralized management of all prompt templates used in the InsightBot.
    """
    
    # System message template
    SYSTEM_MESSAGE = """
    You are InsightBot, an AI assistant specialized in data analysis and visualization. 
    Your purpose is to help users understand their data through clear explanations, 
    insightful visualizations, and actionable insights.
    
    When responding to user queries:
    1. Be concise but thorough in your explanations
    2. Use markdown formatting for better readability
    3. When appropriate, suggest relevant visualizations or analyses
    4. Always consider the context of previous messages
    5. If you're unsure about something, ask clarifying questions
    
    Current configuration:
    - Model: {model}
    - Temperature: {temperature}
    - Max tokens: {max_tokens}
    """
    
    # Default chat prompt
    @classmethod
    def get_chat_prompt(cls, **kwargs) -> ChatPromptTemplate:
        """Get the default chat prompt template."""
        system_template = cls.SYSTEM_MESSAGE.format(
            model=kwargs.get("model", Config.DEFAULT_MODEL),
            temperature=kwargs.get("temperature", Config.TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", Config.MAX_TOKENS)
        )
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
    
    # Data analysis specific prompt
    DATA_ANALYSIS_PROMPT = """
    Analyze the following data and provide insights:
    
    Context:
    - Dataset: {dataset_name}
    - Rows: {num_rows}
    - Columns: {num_columns}
    
    User's question: {user_question}
    
    Available tools:
    {tools}
    
    Please provide a detailed analysis based on the data and the user's question.
    If visualization would help, use the appropriate tool to create it.
    """
    
    # Visualization suggestion prompt
    VISUALIZATION_SUGGESTION = """
    Based on the dataset and the user's question, here are some visualization suggestions:
    
    Data Overview:
    - Numeric columns: {numeric_columns}
    - Categorical columns: {categorical_columns}
    - Date columns: {date_columns}
    
    Suggested visualizations:
    1. {suggestion_1}
    2. {suggestion_2}
    3. {suggestion_3}
    
    Would you like me to create any of these visualizations for you?
    """
    
    # Error handling prompt
    ERROR_HANDLING_PROMPT = """
    I encountered an error while processing your request:
    
    Error: {error_message}
    
    Here's what might have gone wrong:
    - {possible_cause_1}
    - {possible_cause_2}
    
    How would you like to proceed?
    1. Try a different approach
    2. Provide more details about what you're trying to achieve
    3. Cancel the current operation
    """
    
    # Clarification prompt
    CLARIFICATION_PROMPT = """
    I need some clarification to better assist you with your request.
    
    Your original query: "{user_query}"
    
    Could you please provide more details about:
    1. {clarification_point_1}
    2. {clarification_point_2}
    
    This will help me provide a more accurate and helpful response.
    """
    
    @classmethod
    def get_analysis_prompt(cls, dataset_info: Dict[str, Any], tools_info: str) -> str:
        """Get a prompt for data analysis tasks."""
        return cls.DATA_ANALYSIS_PROMPT.format(
            dataset_name=dataset_info.get("name", "unnamed dataset"),
            num_rows=dataset_info.get("num_rows", 0),
            num_columns=len(dataset_info.get("columns", [])),
            user_question=dataset_info.get("question", ""),
            tools=tools_info
        )
    
    @classmethod
    def get_visualization_suggestions(
        cls, 
        numeric_cols: List[str], 
        categorical_cols: List[str],
        date_cols: List[str]
    ) -> str:
        """Get suggestions for visualizations based on column types."""
        suggestions = []
        
        # Generate suggestions based on available columns
        if numeric_cols:
            if len(numeric_cols) >= 2:
                suggestions.append(f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}")
            
            if len(numeric_cols) >= 1 and categorical_cols:
                suggestions.append(f"Box plot of {numeric_cols[0]} by {categorical_cols[0]}")
            
            suggestions.append(f"Histogram of {numeric_cols[0]}")
        
        if categorical_cols:
            if len(categorical_cols) >= 2:
                suggestions.append(f"Stacked bar chart of {categorical_cols[0]} by {categorical_cols[1]}")
            else:
                suggestions.append(f"Bar chart of {categorical_cols[0]}")
        
        if date_cols and numeric_cols:
            suggestions.append(f"Time series plot of {numeric_cols[0]} over {date_cols[0]}")
        
        # Ensure we have at least 3 suggestions
        while len(suggestions) < 3:
            suggestions.append("Pair plot of numeric columns")
        
        return cls.VISUALIZATION_SUGGESTION.format(
            numeric_columns=", ".join(numeric_cols) if numeric_cols else "None",
            categorical_columns=", ".join(categorical_cols) if categorical_cols else "None",
            date_columns=", ".join(date_cols) if date_cols else "None",
            suggestion_1=suggestions[0],
            suggestion_2=suggestions[1],
            suggestion_3=suggestions[2]
        )
