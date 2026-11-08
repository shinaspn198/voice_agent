from groq import AsyncGroq
from app.config import GROQ_API_KEY, GROQ_MODEL

client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are NILA, a helpful and friendly real-time voice assistant. "
    "Give clear, natural, concise responses. "
    "Avoid unnecessary headings and long explanations unless requested."
)


async def generate_response(user_message: str) -> str:
    """Generate a complete response (used by the existing /chat endpoint)."""
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_completion_tokens=500,
    )

    return (response.choices[0].message.content or "").strip()


async def generate_response_stream(user_message: str):
    """Generate a response progressively for the streaming endpoint."""
    stream = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_completion_tokens=500,
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content