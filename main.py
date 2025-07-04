from agent import agent

def run_bot():
    while True:
        query = input("\n🤖 Ask InsightBot: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print(f"🔍 InsightBot: {response}")

run_bot()
