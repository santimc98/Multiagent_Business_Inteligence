import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from src.utils.reviewer_llm import init_reviewer_llm
from src.utils.senior_protocol import SENIOR_EVIDENCE_RULE

load_dotenv()


def _extract_json_object(text: str) -> str | None:
    if not text:
        return None
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_str = False
                escape = False
            continue
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _normalize_reviewer_gate_name(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("name", "id", "gate"):
            value = item.get(key)
            if value:
                return str(value).strip()
        return ""
    if item is None:
        return ""
    return str(item).strip()


def apply_reviewer_gate_filter(result: Dict[str, Any], reviewer_gates: List[Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    allowed = {
        name.lower()
        for name in (_normalize_reviewer_gate_name(g) for g in (reviewer_gates or []))
        if name
    }
    if allowed:
        filtered = []
        for g in result.get("failed_gates", []):
            gate_name = _normalize_reviewer_gate_name(g)
            if gate_name.lower() in allowed:
                filtered.append(gate_name)
        result["failed_gates"] = filtered
        if result.get("status") == "REJECTED" and not result.get("failed_gates"):
            result["status"] = "APPROVE_WITH_WARNINGS"
            result["feedback"] = "Spec-driven gating: no reviewer gates failed; downgraded to warnings."
    return result

class ReviewerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Reviewer Agent with MIMO v2 Flash.
        """
        self.provider, self.client, self.model_name, self.model_warning = init_reviewer_llm(api_key)
        if self.model_warning:
            print(f"WARNING: {self.model_warning}")
        self.last_prompt = None
        self.last_response = None

    def review_code(
        self,
        code: str,
        analysis_type: str = "predictive",
        business_objective: str = "",
        strategy_context: str = "",
        evaluation_spec: Dict[str, Any] | None = None,
        reviewer_view: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        
        output_format_instructions = """
        Return a raw JSON object:
        {
            "status": "APPROVED" | "APPROVE_WITH_WARNINGS" | "REJECTED",
            "feedback": "Detailed explanation of what to fix if rejected, or 'Looks good' if approved.",
            "failed_gates": ["List", "of", "failed", "principles"],
            "required_fixes": ["List", "of", "specific", "instructions", "for", "the", "engineer"],
            "evidence": [
                {"claim": "Short claim", "source": "artifact_path#key_or_script_path:line or missing"}
            ]
        }
        """

        from src.utils.prompting import render_prompt

        reviewer_view = reviewer_view or {}
        eval_spec_json = json.dumps(evaluation_spec or {}, indent=2)
        reviewer_gates = []
        if isinstance(evaluation_spec, dict):
            reviewer_gates = evaluation_spec.get("reviewer_gates") or evaluation_spec.get("gates") or []
        view_gates = reviewer_view.get("reviewer_gates")
        if isinstance(view_gates, list) and view_gates:
            reviewer_gates = view_gates
        allowed_columns = []
        if isinstance(evaluation_spec, dict):
            for key in ("allowed_columns", "canonical_columns", "required_columns", "contract_columns"):
                cols = evaluation_spec.get(key)
                if isinstance(cols, list) and cols:
                    allowed_columns = [str(c) for c in cols if c]
                    break
        if isinstance(reviewer_view.get("required_outputs"), list) and reviewer_view.get("required_outputs"):
            allowed_columns = allowed_columns or []
        strategy_summary = reviewer_view.get("strategy_summary") or strategy_context
        objective_type = reviewer_view.get("objective_type") or analysis_type
        expected_metrics = reviewer_view.get("expected_metrics") or []
        execution_diagnostics = (
            reviewer_view.get("execution_diagnostics")
            if isinstance(reviewer_view, dict)
            else None
        )
        if not isinstance(execution_diagnostics, dict):
            execution_diagnostics = (
                evaluation_spec.get("execution_diagnostics")
                if isinstance(evaluation_spec, dict)
                else {}
            )
        if not isinstance(execution_diagnostics, dict):
            execution_diagnostics = {}

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Senior Technical Lead and Security Auditor.

        === EVIDENCE RULE ===
        $senior_evidence_rule
        
        CONTEXT: 
        - Objective Type: "$analysis_type"
        - Strategy Summary: "$strategy_context"
        - Evaluation Spec (JSON): $evaluation_spec_json
        - Reviewer View (JSON): $reviewer_view_json
        - Reviewer Gates (only these can fail): $reviewer_gates
        - Allowed Columns (if provided): $allowed_columns_json
        - Expected Metrics (if provided): $expected_metrics_json
        - Execution Diagnostics (JSON): $execution_diagnostics_json
        
        ### CRITERIA FOR APPROVAL (QUALITY FIRST PRINCIPLES)

        1. **SECURITY & SAFETY (Non-Negotiable):**
           - No malicious code, no external network calls (except sanctioned APIs), no file system deletions outside `data/`.
           
        2. **METHODOLOGY VERIFICATION (Results-Based, Not Syntax-Based):**
           - **Baseline Check:** Does the code establish a baseline? (e.g., Dummy Classifier)? PREFERRED but not required.
           - **Validation Rigor:**
             * WARN (not REJECT) if a predictive model uses a single `train_test_split` on a small dataset.
             * Cross-Validation is PREFERRED but holdout is ACCEPTABLE if metrics.json shows reasonable results.
             * REJECT only if testing is done on Training Data (actual Leakage detected in results).
           - **Assumption Check:** Does the code handle critical assumptions? (WARN if missing, REJECT only if results are clearly invalid).
           - **EXECUTION-AWARE PRINCIPLE:** If the code produces valid metrics.json and alignment_check.json,
             the methodology is likely sound. Trust execution results over static code patterns.
           
        3. **BUSINESS VALUE CHECK (The "So What?"):**
           - **Alignment:** Does this analysis *actually* answer: "$business_objective"?
           - *Example:* If objective is "Explain Drivers", a "Black Box" Neural Net is bad. A Decision Tree or Regression is better.
           - *Action:* If the model choice contradicts the objective's need for explainability vs. performance, REJECT it with this specific feedback.

        4. **ENGINEERING STANDARDS:**
           - **Robustness:** Will this crash on empty inputs? (e.g., `df.empty` checks).
           - **Modernity:** No deprecated library calls (e.g., `use_label_encoder` in XGBoost).
           - **Cleanliness:** Code must be syntactically correct and runnable.

        5. **COLUMN MAPPING INTEGRITY:**
           - If Allowed Columns are provided, do NOT use hardcoded column names outside that list.
           - If Allowed Columns are missing, flag hardcoded columns as WARNING only (do not reject).
        
        ### VERDICT LOGIC
        - **REJECT**: Critical Security Violations, Data Leakage, Wrong Method (Regression for Classification), Syntax Errors, Missing Imports.
        - **APPROVE_WITH_WARNINGS**: Minor issues (e.g. suboptimal parameter, messy comments, slight style deviations) that do NOT affect correctness or safety.
        - **APPROVED**: Code is clean, safe, and correct.

        ### SPEC-DRIVEN EVALUATION (MANDATORY)
        - Only fail gates that appear in Reviewer Gates.
        - If a rule is NOT present in Reviewer Gates, you may mention it as a warning but MUST NOT reject for it.
        - If Reviewer Gates is empty, fall back to the general criteria but prefer APPROVE_WITH_WARNINGS when uncertain.

        ### EVIDENCE REQUIREMENT
        - Any REJECT or warning must cite evidence from the provided artifacts or code.
        - Include evidence in feedback using: EVIDENCE: <artifact_path>#<key> -> <short snippet>
        - If you cannot find evidence, downgrade to APPROVE_WITH_WARNINGS and state NO_EVIDENCE_FOUND.
        - SELF-CHECK BEFORE REJECT: without at least one concrete evidence item, you must not reject.
        - Populate the "evidence" list with 3-8 items. If evidence is missing, use source="missing".
        - Evidence sources must be artifact paths or script paths; otherwise use source="missing".

        ### OUTPUT FORMAT
        $output_format_instructions
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            analysis_type=str(objective_type).upper(),
            business_objective=business_objective,
            strategy_context=strategy_summary,
            evaluation_spec_json=eval_spec_json,
            reviewer_view_json=json.dumps(reviewer_view, indent=2),
            reviewer_gates=reviewer_gates,
            allowed_columns_json=json.dumps(allowed_columns, indent=2),
            expected_metrics_json=json.dumps(expected_metrics, indent=2),
            execution_diagnostics_json=json.dumps(execution_diagnostics, indent=2),
            output_format_instructions=output_format_instructions,
            senior_evidence_rule=SENIOR_EVIDENCE_RULE,
        )
        
        USER_PROMPT_TEMPLATE = "REVIEW THIS CODE:\n\n$code"
        user_prompt = render_prompt(USER_PROMPT_TEMPLATE, code=code)
        self.last_prompt = system_prompt + "\n\n" + user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if not self.client or self.provider == "none":
            return {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": "Reviewer LLM disabled; continuing with deterministic evidence only.",
                "failed_gates": [],
                "required_fixes": [],
            }

        try:
            if self.provider == "gemini":
                print(f"DEBUG: Reviewer calling Gemini ({self.model_name})...")
                response = self.client.generate_content(system_prompt + "\n\n" + user_prompt)
                content = response.text
            else:
                print(f"DEBUG: Reviewer calling MIMO ({self.model_name})...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={'type': 'json_object'},
                    temperature=0.0
                )
                content = response.choices[0].message.content
            self.last_response = content
            result = self._parse_json_payload(content)
            
            # Normalize lists
            for field in ['failed_gates', 'required_fixes']:
                val = result.get(field, [])
                if isinstance(val, str):
                    result[field] = [val]
                elif not isinstance(val, list):
                    result[field] = []
                else:
                    result[field] = val

            result = apply_reviewer_gate_filter(result, reviewer_gates)
            return result

        except Exception as e:
            # Fail-open: rely on deterministic gates for blocking.
            print(f"Reviewer API Error: {e}")
            return {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": (
                    f"Reviewer unavailable (API error: {e}). "
                    "Continuing with deterministic evidence only."
                ),
                "failed_gates": [],
                "required_fixes": [],
            }

    def _clean_json(self, text: str) -> str:
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()

    def _parse_json_payload(self, text: str) -> Dict[str, Any]:
        parse_error: Exception | None = None
        cleaned = self._clean_json(text or "")
        candidates = [cleaned, _extract_json_object(cleaned), _extract_json_object(text or "")]
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            blob = candidate.strip()
            if not blob or blob in seen:
                continue
            seen.add(blob)
            try:
                parsed = json.loads(blob)
                if isinstance(parsed, dict):
                    return parsed
                parse_error = ValueError("Reviewer JSON payload is not an object")
            except Exception as err:
                parse_error = err
        if parse_error:
            raise parse_error
        raise ValueError("Empty reviewer JSON payload")

    def evaluate_results(
        self,
        execution_output: str,
        business_objective: str,
        strategy_context: str,
        evaluation_spec: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluates execution results.
        Phase 1: Deterministic Runtime Error Triage (No LLM).
        Phase 2: Semantic LLM Evaluation (Only if no errors).
        """
        
        # --- PHASE 1: DETERMINISTIC PRE-TRIAGE ---
        # Check for Tracebacks or Execution Errors
        if "Traceback (most recent call last)" in execution_output or "EXECUTION ERROR" in execution_output:
            print("Reviewer: Runtime Error Detected. Skipping LLM eval.")
            
            # Specific Fixes for Common Errors
            failed_gates = ["Runtime Correctness"]
            required_fixes = []
            
            # 1. String to Float Conversion Error (Common in Correlation/ROC with dirty data)
            if "could not convert string to float" in execution_output:
                required_fixes.append("Convert target/features to numeric using `pd.to_numeric(..., errors='coerce')` or `df.factorize()`.")
                required_fixes.append("Map binary strings (yes/no) to 0/1 ensuring `map({'yes':1, 'no':0})` handles case.")
                required_fixes.append("Drop non-numeric columns before Correlation Matrix.")
            
            # 2. General Traceback Fallback
            if not required_fixes:
                required_fixes.append("Fix the Python Runtime Error shown in the logs.")
                required_fixes.append("Wrap dangerous blocks (like plotting or modeling) in try/except.")

            return {
                "status": "NEEDS_IMPROVEMENT",
                "feedback": f"Runtime Error detected in execution. Fix the code to run successfully.\nError Snippet: {execution_output[-500:]}",
                "failed_gates": failed_gates,
                "required_fixes": required_fixes,
                "retry_worth_it": True
            }

        # --- PHASE 2: LLM SEMANTIC EVALUATION ---
        output_format_instructions = """
        Return a raw JSON object:
        {
            "status": "APPROVED" | "NEEDS_IMPROVEMENT",
            "feedback": "Specific instructions for the ML Engineer.",
            "failed_gates": [],
            "required_fixes": [],
            "retry_worth_it": true | false,
            "evidence": [
                {"claim": "Short claim", "source": "artifact_path#key_or_script_path:line or missing"}
            ]
        }
        """

        # Truncate Output for Token Safety
        truncated_output = execution_output[-4000:] if len(execution_output) > 4000 else execution_output

        from src.utils.prompting import render_prompt
        
        eval_spec_json = json.dumps(evaluation_spec or {}, indent=2)

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Senior Data Science Lead.
        Your goal is to evaluate the RESULTS of an analysis against the Business Objective.

        === EVIDENCE RULE ===
        $senior_evidence_rule
        
        *** BUSINESS OBJECTIVE ***
        "$business_objective"
        
        *** STRATEGY CONTEXT ***
        $strategy_context

        *** EVALUATION SPEC (JSON) ***
        $evaluation_spec_json
        
        *** EXECUTION OUTPUT (Truncated) ***
        $truncated_output
        
        *** EVALUATION CRITERIA (BUSINESS-FIRST) ***
        1. **Answer Quality:** Does the output provide a clear answer/insight relevant to the BUSINESS OBJECTIVE?
        2. **Visuals:** Are plots generated when required? (If required by spec but missing, flag as warning).
        3. **Metrics - BUSINESS-RELATIVE EVALUATION (CRITICAL):**
           - DO NOT use arbitrary fixed thresholds (e.g., "Accuracy > 0.5").
           - INSTEAD, evaluate metrics RELATIVE TO:
             a) The BASELINE model (if provided): Is the final model better than the baseline?
             b) The PROBLEM DIFFICULTY: Imbalanced classes, noisy data, or limited features justify lower scores.
             c) The BUSINESS VALUE: A 60% accuracy model that identifies 3x more leads than random is valuable.
           - LOW ABSOLUTE SCORES CAN BE VALUABLE if they represent meaningful improvement over baseline.
           - Example: AUC=0.65 on a 1% positive class is excellent; AUC=0.65 on balanced data is mediocre.
        4. **Validation:** If predictive, was validation performed? (Cross-validation preferred but holdout acceptable).
        5. **Safety Outputs:** Does alignment_check.json exist? (Execution-aware check, not code pattern matching).
        - Only enforce criteria that are required by the Evaluation Spec.

        *** SENIOR TECH LEAD MINDSET ***
        You care that the system WORKS, is SAFE, and SOLVES THE PROBLEM.
        You do NOT care if the code looks exactly like a textbook example.

        *** FALLBACK LOGIC ***
        - If results are "weak" in absolute terms but methodology is sound and improves over baseline => APPROVE with note.
        - REJECT only if there are specific TECHNICAL fixes that would materially improve results.
        - If "Traceback" or "Error" persists in output despite your checks, REJECT.

        *** EVIDENCE REQUIREMENT ***
        - Any NEEDS_IMPROVEMENT or warning must cite evidence from artifacts or execution output.
        - Include evidence in feedback using: EVIDENCE: <artifact_path>#<key> -> <short snippet>
        - If you cannot find evidence, downgrade to APPROVE_WITH_WARNINGS and state NO_EVIDENCE_FOUND.
        - SELF-CHECK BEFORE REJECT: without at least one concrete evidence item, you must not reject.
        - Populate the "evidence" list with 3-8 items. If evidence is missing, use source="missing".
        - Evidence sources must be artifact paths or script paths; otherwise use source="missing".

        OUTPUT FORMAT (JSON):
        $output_format_instructions
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=business_objective,
            strategy_context=strategy_context,
            truncated_output=truncated_output,
            evaluation_spec_json=eval_spec_json,
            output_format_instructions=output_format_instructions,
            senior_evidence_rule=SENIOR_EVIDENCE_RULE,
        )
        self.last_prompt = system_prompt + "\n\nEvaluate results."

        if not self.client or self.provider == "none":
            return {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": "Reviewer LLM disabled; continuing with deterministic evidence only.",
                "failed_gates": [],
                "required_fixes": [],
                "retry_worth_it": False,
            }

        try:
            if self.provider == "gemini":
                print(f"DEBUG: Reviewer evaluation calling Gemini ({self.model_name})...")
                response = self.client.generate_content(system_prompt + "\n\nEvaluate results.")
                content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Evaluate results."}
                    ],
                    response_format={'type': 'json_object'},
                    temperature=0.1
                )
                content = response.choices[0].message.content
            self.last_response = content
            result = self._parse_json_payload(content)
            
            # Defaults for backward compatibility
            if "failed_gates" not in result: result["failed_gates"] = []
            if "required_fixes" not in result: result["required_fixes"] = []
            if "retry_worth_it" not in result: result["retry_worth_it"] = True
            
            return result
            
        except Exception as e:
            print(f"Reviewer Evaluation Error: {e}")
            return {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": (
                    f"Evaluation skipped due to error: {e}. "
                    "Continuing with deterministic evidence only."
                ),
                "failed_gates": [],
                "required_fixes": [],
                "retry_worth_it": False
            }
