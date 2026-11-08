
from groq import AsyncGroq
from app.config import GROQ_API_KEY

client = AsyncGroq(api_key=GROQ_API_KEY)


async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    response = await client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="json"
    )

    return response.text
