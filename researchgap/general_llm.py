from __future__ import annotations

import time
from typing import Any, Dict

from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError


def call_chat_completion(config: Any, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> Dict[str, Any]:
    if not config.has_llm:
        raise ValueError("No API key or model is configured. Add LLM_API_KEY and LLM_MODEL before running a real experiment.")

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        usage_payload = {}
        if usage is not None:
            usage_payload = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return {
            "answer": answer,
            "model_name": config.model_name,
            "execution_time": round(time.time() - start, 2),
            "usage": usage_payload,
            "status": "success",
        }
    except AuthenticationError as exc:
        raise ValueError(f"Authentication failed: {exc}") from exc
    except RateLimitError as exc:
        raise ValueError(f"Rate limit exceeded: {exc}") from exc
    except APIConnectionError as exc:
        raise ValueError(f"Connection error: {exc}") from exc
    except APIStatusError as exc:
        raise ValueError(f"Model request failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"LLM execution failed: {exc}") from exc


def run_general_llm(question: str, document_text: str, config: Any) -> Dict[str, Any]:
    system_prompt = (
        "You are a careful general-purpose scientific assistant. Answer the question using the supplied document text "
        "without inventing facts or citations. If the text does not provide enough evidence, say so clearly."
    )
    user_prompt = (
        f"Research question:\n{question}\n\n"
        f"Uploaded document text:\n{document_text[:20000]}"
    )
    return call_chat_completion(config, system_prompt, user_prompt, max_tokens=700)
