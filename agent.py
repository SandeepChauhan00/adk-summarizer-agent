from google.adk.agents import Agent

summarizer_agent = Agent(
    name="summarizer_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a precise text-summarization agent.\n"
        "Rules:\n"
        "1. Read the user-provided text carefully.\n"
        "2. Return ONLY a concise summary (2-3 sentences).\n"
        "3. Do NOT add opinions or extra formatting.\n"
        "4. If input is very short, return it as-is."
    ),
)
