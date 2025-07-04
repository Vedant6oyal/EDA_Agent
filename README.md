# InsightBot

InsightBot is a powerful data analysis assistant that helps you explore, visualize, and gain insights from your datasets using natural language. Built with Python, LangChain, and DuckDB, it provides an intuitive interface for data analysis tasks.

## Features

- **Natural Language Processing**: Interact with your data using plain English
- **Data Visualization**: Generate various plots and charts with simple commands
- **Statistical Analysis**: Get detailed statistical summaries of your data
- **Conversational Interface**: Maintains context across multiple queries
- **Easy to Use**: No coding required for basic data exploration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/insightbot.git
   cd insightbot
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Place your CSV file in the project directory (or use the provided `snapshots_2000.csv`)

2. Run the InsightBot:
   ```bash
   python main.py
   ```

3. Start asking questions about your data, for example:
   - "Show me a histogram of revenue"
   - "What are the average expenses by category?"
   - "Create a scatter plot of revenue vs. profit"
   - "What's the correlation between revenue and expenses?"

## Project Structure

```
insightbot/
├── main.py                  # Entry point
├── data_loader.py           # CSV/DuckDB handling
├── agent.py                 # LangChain agent logic
├── tools/                   # Custom tools
│   ├── plot_tool.py         # Data visualization
│   └── stats_tool.py        # Statistical analysis
├── memory.py                # Memory management
├── prompts.py               # Prompt templates
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── snapshots_2000.csv            # Sample dataset
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Uses [DuckDB](https://duckdb.org/) for fast in-memory analytics
- Powered by [OpenAI](https://openai.com/)'s language models
