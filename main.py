import os
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent import summarizer_agent

app = FastAPI(title="ADK Text Summarizer Agent")

APP_NAME = "summarizer_app"
session_service = InMemorySessionService()
runner = Runner(
    agent=summarizer_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)

class SummarizeResponse(BaseModel):
    summary: str
    agent: str = "summarizer_agent"
    model: str = "gemini-2.0-flash"

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=user_id,
    )
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Please summarize:\n\n{req.text}")],
    )
    summary_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=user_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                summary_text = event.content.parts[0].text
            break
    if not summary_text:
        raise HTTPException(status_code=500, detail="Empty response")
    return SummarizeResponse(summary=summary_text)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"service": "ADK Summarizer", "usage": "POST /summarize"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
