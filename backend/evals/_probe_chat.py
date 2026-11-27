"""One JSON chat call to confirm the eval model returns parseable items."""

import json

from openai import OpenAI

from app.config import NVIDIA_BASE_URL, settings


def main() -> None:
    client = OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=NVIDIA_BASE_URL,
        timeout=60.0,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=settings.nvidia_chat_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    '{"task":"Write one question about this Apple fiscal 2024 sentence. '
                    'Copy figures exactly. JSON shape {\\"items\\":[{\\"id\\":\\"1\\",'
                    '\\"question\\":\\"\\",\\"expected_answer\\":\\"\\",\\"claims\\":[]}]} '
                    'Sentence: The Company believes its balances of unrestricted cash, '
                    "cash equivalents and marketable securities, which totaled $140.8 billion "
                    'as of September 28, 2024, will be sufficient."}'
                ),
            },
        ],
    )
    text = response.choices[0].message.content or ""
    print(text[:1500])
    print("---")
    print(json.loads(text)["items"][0]["question"])


if __name__ == "__main__":
    main()
