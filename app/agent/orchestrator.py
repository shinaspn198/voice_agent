from app.agent.llm import generate_response, generate_response_stream


async def process_message(user_message: str) -> str:
    """Process a normal, non-streaming chat message."""
    if not user_message.strip():
        return "Please enter a message."

    return await generate_response(user_message)


async def process_message_stream(user_message: str):
    """Process a message and yield response chunks."""
    if not user_message.strip():
        yield "Please enter a message."
        return

    async for chunk in generate_response_stream(user_message):
        yield chunk