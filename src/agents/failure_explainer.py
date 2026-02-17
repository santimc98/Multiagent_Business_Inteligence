import os
from typing import Any, Dict

from dotenv import load_dotenv

from src.utils.retries import call_with_retries

load_dotenv()


class FailureExplainerAgent:
    """
    Explains runtime failures using code + traceback + context.
    Returns a short, plain-text diagnosis to feed back into the next attempt.
    Uses the Gemini Flash API (same as reviewers).
    """

    def __init__(self, api_key: Any = None):
        self._gemini_model = None
        self._model_name = None

        google_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            try:
                import google.generativeai as genai
                from google.generativeai.types import HarmCategory, HarmBlockThreshold

                genai.configure(api_key=google_key)
                self._model_name = os.getenv(
                    "FAILURE_EXPLAINER_MODEL", "gemini-3-flash-preview"
                )
                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                self._gemini_model = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
            except Exception:
                self._gemini_model = None

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini and return the text response."""
        response = self._gemini_model.generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            candidates = getattr(response, "candidates", None)
            if candidates:
                content = getattr(candidates[0], "content", None)
                if content and getattr(content, "parts", None):
                    text = content.parts[0].text
        return (text or "").strip()

    def explain_data_engineer_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        if not code or not error_details:
            return ""
        if not self._gemini_model:
            return self._fallback(error_details)

        ctx = context or {}
        code_snippet = self._truncate(code, 6000)
        error_snippet = self._truncate(error_details, 4000)
        context_snippet = self._truncate(str(ctx), 2000)

        prompt = (
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
            "Do NOT include code. Do NOT restate the full traceback.\n\n"
            f"CODE:\n{code_snippet}\n\n"
            f"ERROR:\n{error_snippet}\n\n"
            f"CONTEXT:\n{context_snippet}\n"
        )

        try:
            content = call_with_retries(lambda: self._call_gemini(prompt), max_retries=2)
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
        if not self._gemini_model:
            return self._fallback(error_details)

        ctx = context or {}
        code_snippet = self._truncate(code, 6000)
        error_snippet = self._truncate(error_details, 4000)
        context_snippet = self._truncate(str(ctx), 2000)

        prompt = (
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
            "Do NOT include code. Do NOT restate the full traceback.\n\n"
            f"CODE:\n{code_snippet}\n\n"
            f"ERROR:\n{error_snippet}\n\n"
            f"CONTEXT:\n{context_snippet}\n"
        )

        try:
            content = call_with_retries(lambda: self._call_gemini(prompt), max_retries=2)
        except Exception:
            return self._fallback(error_details)

        return (content or "").strip()

    def _fallback(self, error_details: str) -> str:
        if not error_details:
            return ""
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
