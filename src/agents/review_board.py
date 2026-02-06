import json
import re
from typing import Any, Dict, List

from src.utils.reviewer_llm import init_reviewer_llm
from src.utils.senior_protocol import SENIOR_EVIDENCE_RULE


class ReviewBoardAgent:
    """
    Final adjudicator for reviewer outputs.
    Consolidates Reviewer, QA Reviewer, and Results Advisor findings.
    """

    def __init__(self, api_key: str = None):
        self.provider, self.client, self.model_name, self.model_warning = init_reviewer_llm(api_key)
        if self.model_warning:
            print(f"WARNING: {self.model_warning}")
        self.last_prompt = None
        self.last_response = None

    def adjudicate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context = context or {}
        if not self.client or self.provider == "none":
            return self._fallback(context)

        output_format = """
        Return a raw JSON object:
        {
          "status": "APPROVED" | "APPROVE_WITH_WARNINGS" | "NEEDS_IMPROVEMENT" | "REJECTED",
          "summary": "Single concise paragraph.",
          "failed_areas": ["reviewer_alignment", "qa_gates", "results_quality"],
          "required_actions": ["action 1", "action 2"],
          "confidence": "high" | "medium" | "low",
          "evidence": [
            {"claim": "Short claim", "source": "artifact_path#key_or_script_path:line or missing"}
          ]
        }
        """
        system_prompt = (
            "You are the Review Board for a multi-agent ML system.\n"
            "Your job is to issue the final verdict using the evidence from:\n"
            "- Reviewer (strategy/contract/code alignment)\n"
            "- QA Reviewer (universal + contract QA gates)\n"
            "- Results Advisor (quality and improvement potential)\n\n"
            "=== EVIDENCE RULE ===\n"
            f"{SENIOR_EVIDENCE_RULE}\n\n"
            "Decision policy:\n"
            "1) If there are unresolved hard failures with high confidence, return REJECTED.\n"
            "2) If fixes are required but resolvable in next iteration, return NEEDS_IMPROVEMENT.\n"
            "3) If results are usable but with caveats, return APPROVE_WITH_WARNINGS.\n"
            "4) If all critical areas pass, return APPROVED.\n"
            "Do not invent evidence.\n\n"
            "Output format:\n"
            f"{output_format}"
        )
        user_prompt = "REVIEW_STACK_CONTEXT:\n" + json.dumps(context, ensure_ascii=True, indent=2)
        self.last_prompt = system_prompt + "\n\n" + user_prompt

        try:
            if self.provider == "gemini":
                response = self.client.generate_content(self.last_prompt)
                content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content
            self.last_response = content
            parsed = json.loads(self._clean_json(content))
            return self._normalize(parsed, context)
        except Exception:
            return self._fallback(context)

    def _fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        qa = context.get("qa_reviewer") if isinstance(context.get("qa_reviewer"), dict) else {}
        reviewer = context.get("reviewer") if isinstance(context.get("reviewer"), dict) else {}
        evaluator = context.get("result_evaluator") if isinstance(context.get("result_evaluator"), dict) else {}
        runtime = context.get("runtime") if isinstance(context.get("runtime"), dict) else {}

        statuses = [
            str(qa.get("status", "")).upper(),
            str(reviewer.get("status", "")).upper(),
            str(evaluator.get("status", "")).upper(),
        ]
        hard_failures = []
        for payload in (qa, reviewer, evaluator):
            hf = payload.get("hard_failures")
            if isinstance(hf, list):
                hard_failures.extend(str(x) for x in hf if x)

        if hard_failures:
            status = "REJECTED" if runtime.get("runtime_fix_terminal") else "NEEDS_IMPROVEMENT"
        elif "NEEDS_IMPROVEMENT" in statuses or "REJECTED" in statuses:
            status = "NEEDS_IMPROVEMENT"
        elif "APPROVE_WITH_WARNINGS" in statuses:
            status = "APPROVE_WITH_WARNINGS"
        else:
            status = "APPROVED"
        return {
            "status": status,
            "summary": "Fallback board verdict from reviewer packets.",
            "failed_areas": ["qa_gates"] if hard_failures else [],
            "required_actions": ["Apply reviewer-required fixes and rerun."] if status in {"NEEDS_IMPROVEMENT", "REJECTED"} else [],
            "confidence": "medium",
            "evidence": [],
        }

    def _normalize(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        status = str(payload.get("status", "")).strip().upper()
        if status not in {"APPROVED", "APPROVE_WITH_WARNINGS", "NEEDS_IMPROVEMENT", "REJECTED"}:
            status = self._fallback(context)["status"]
        failed_areas = payload.get("failed_areas")
        required_actions = payload.get("required_actions")
        evidence = payload.get("evidence")
        return {
            "status": status,
            "summary": str(payload.get("summary", "")).strip(),
            "failed_areas": [str(x) for x in failed_areas] if isinstance(failed_areas, list) else [],
            "required_actions": [str(x) for x in required_actions] if isinstance(required_actions, list) else [],
            "confidence": str(payload.get("confidence", "medium")).lower() if payload.get("confidence") else "medium",
            "evidence": evidence if isinstance(evidence, list) else [],
        }

    def _clean_json(self, text: str) -> str:
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)
        return text.strip()

