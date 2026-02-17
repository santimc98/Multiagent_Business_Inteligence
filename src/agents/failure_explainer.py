import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.retries import call_with_retries

load_dotenv()


class FailureExplainerAgent:
    """
    Explains runtime failures using code + traceback + context.
    Returns a short, plain-text diagnosis to feed back into the next attempt.
    """

    def __init__(self, api_key: Any = None):
        self.api_key = api_key or os.getenv("MIMO_API_KEY")
        self.client = None
        self.model_name = "mimo-v2-flash"
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.xiaomimimo.com/v1",
                timeout=None,
            )
        # Fallback: use OpenRouter if MIMO is unavailable
        self._fallback_client = None
        self._fallback_model = None
        if not self.client:
            _or_key = os.getenv("OPENROUTER_API_KEY")
            if _or_key:
                self._fallback_client = OpenAI(
                    api_key=_or_key,
                    base_url="https://openrouter.ai/api/v1",
                    timeout=60.0,
                )
                self._fallback_model = os.getenv(
                    "FAILURE_EXPLAINER_FALLBACK_MODEL", "google/gemini-2.0-flash-001"
                )

    def _get_active_client_and_model(self):
        """Return (client, model_name) using primary or fallback."""
        if self.client:
            return self.client, self.model_name
        if self._fallback_client:
            return self._fallback_client, self._fallback_model
        return None, None

    def explain_data_engineer_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        if not code or not error_details:
            return ""
        active_client, active_model = self._get_active_client_and_model()
        if not active_client:
            return self._fallback(error_details)

        ctx = context or {}
        code_snippet = self._truncate(code, 6000)
        error_snippet = self._truncate(error_details, 4000)
        context_snippet = self._truncate(str(ctx), 2000)

        system_prompt = (
            "You are a senior debugging assistant. "
            "Given the generated Python cleaning code, the traceback/error, and context, "
            "explain why the failure happened and how to fix it. "
            "Return concise plain text (3-6 short lines). "
            "Use this format with short lines: "
            "WHERE: <location or step>, WHY: <root cause>, FIX: <specific change>. "
            "Prioritize the earliest root cause, not just the final exception. "
            "Be concrete about the coding invariant that was violated (shape/length mismatch, "
            "missing columns, incorrect file path, stale artifacts, wrong import, etc.). "
            "If uncertain, propose a minimal diagnostic check to confirm the cause. "
            "Do NOT include code. Do NOT restate the full traceback."
        )
        user_prompt = (
            "CODE:\n"
            + code_snippet + "\n\n"
            "ERROR:\n"
            + error_snippet + "\n\n"
            "CONTEXT:\n"
            + context_snippet + "\n"
        )

        def _call_model():
            response = active_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content

        try:
            content = call_with_retries(_call_model, max_retries=2)
        except Exception:
            return self._fallback(error_details)

        return (content or "").strip()

    def explain_ml_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        if not code or not error_details:
            return ""
        active_client, active_model = self._get_active_client_and_model()
        if not active_client:
            return self._fallback(error_details)

        ctx = context or {}
        code_snippet = self._truncate(code, 6000)
        error_snippet = self._truncate(error_details, 4000)
        context_snippet = self._truncate(str(ctx), 2000)

        system_prompt = (
            "You are a senior ML debugging assistant. "
            "Given the generated ML Python code, the runtime error output, and context, "
            "explain why the failure happened and how to fix it. "
            "Return concise plain text (3-6 short lines). "
            "Use this format with short lines: "
            "WHERE: <location or step>, WHY: <root cause>, FIX: <specific change>. "
            "Prioritize the earliest root cause, not just the final exception. "
            "Name the violated invariant (mismatched shapes/lengths, pipeline refit side effects, "
            "missing column, wrong import, wrong file path, or derived field not created). "
            "If multiple errors appear, address the first causal one. "
            "If uncertain, propose a minimal diagnostic check to confirm the cause. "
            "Do NOT include code. Do NOT restate the full traceback."
        )
        user_prompt = (
            "CODE:\n"
            + code_snippet + "\n\n"
            "ERROR:\n"
            + error_snippet + "\n\n"
            "CONTEXT:\n"
            + context_snippet + "\n"
        )

        def _call_model():
            response = active_client.chat.completions.create(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content

        try:
            content = call_with_retries(_call_model, max_retries=2)
        except Exception:
            return self._fallback(error_details)

        return (content or "").strip()

    def _fallback(self, error_details: str) -> str:
        if not error_details:
            return ""
        # Provide the raw error as-is; LLM-based diagnosis was unavailable.
        return f"Automated diagnosis unavailable. Raw error summary: {error_details[:500]}"

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-(limit // 2) :]
        return f"{head}\n...[truncated]...\n{tail}"
