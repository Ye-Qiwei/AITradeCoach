from __future__ import annotations

from pydantic import BaseModel

from ai_trading_coach.config import get_settings
from ai_trading_coach.llm.langchain_chat_model import build_langchain_chat_model


class TinySchema(BaseModel):
    ok: bool
    summary: str


def main() -> None:
    settings = get_settings()

    print("=== OpenAI Health Check ===")
    print(f"provider={settings.llm_provider_name!r}")
    print(f"model={settings.selected_llm_model()!r}")
    print(f"timeout={settings.llm_timeout_seconds}")
    print(f"api_key_present={bool(settings.openai_api_key.strip())}")

    settings.validate_llm_or_raise()

    model = build_langchain_chat_model(settings=settings)

    print("\n[1] text invoke")
    text_resp = model.invoke(
        [
            {
                "role": "system",
                "content": "You are a concise assistant.",
            },
            {
                "role": "user",
                "content": "Reply with exactly: OPENAI_TEXT_OK",
            },
        ]
    )
    print("text response:")
    print(text_resp.content)

    print("\n[2] structured invoke")
    structured_model = model.with_structured_output(TinySchema)
    structured_resp = structured_model.invoke(
        [
            {
                "role": "system",
                "content": "Return structured output that matches the schema.",
            },
            {
                "role": "user",
                "content": "Say the API is healthy in one short sentence.",
            },
        ]
    )
    print("structured response:")
    print(structured_resp)

    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()