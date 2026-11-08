import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.orchestrator import (
    process_message,
    process_message_stream,
)
from app.voice.stt import transcribe_audio


app = FastAPI(title="VoxAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "message": "Voice Agent Backend is running!",
    }


# Existing endpoint: keep it for compatibility.
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    ai_reply = await process_message(request.message)
    return {"reply": ai_reply}


# New endpoint: streams response chunks as NDJSON.
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )

    async def generate():
        try:
            async for chunk in process_message_stream(request.message):
                yield json.dumps(
                    {"token": chunk},
                    ensure_ascii=False,
                ) + "\n"

            yield json.dumps({"done": True}) + "\n"

        except Exception as error:
            # Send an error record if generation fails after streaming begins.
            yield json.dumps(
                {"error": str(error)},
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="No audio file received",
        )

    try:
        text = await transcribe_audio(
            audio_bytes,
            file.filename or "recording.webm",
        )

        return {"text": text}

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(error)}",
        )