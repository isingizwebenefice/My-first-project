
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

app = FastAPI(title="NOLAN AI")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

SYSTEM_PROMPT = """
You are NOLAN, a helpful personal AI assistant.

Your responsibilities:
- Answer questions clearly.
- Help with programming and ICT.
- Explain difficult topics simply.
- Help with learning and creativity.
- Be polite, friendly and honest.
- Never pretend to know something you do not know.
"""

class ChatRequest(BaseModel):
    messages: list[dict]

@app.get("/")
async def home():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat(request: ChatRequest):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Keep the latest conversation messages
    for message in request.messages[-20:]:
        if message.get("role") in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": str(message.get("content", ""))
            })

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False
                }
            )

            response.raise_for_status()
            data = response.json()

        return {
            "reply": data["message"]["content"]
        }

    except httpx.ConnectError:
        return {
            "error": "Cannot connect to Ollama. Make sure Ollama is running."
        }

    except httpx.HTTPStatusError as error:
        return {
            "error": f"AI server error: {error.response.text[:300]}"
        }

    except Exception as error:
        return {
            "error": str(error)
        }

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
