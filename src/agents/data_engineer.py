import os
import re
import ast
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.utils.static_safety_scan import scan_code_safety
from src.utils.code_extract import extract_code_block
from src.utils.senior_protocol import SENIOR_ENGINEERING_PROTOCOL
from src.utils.contract_accessors import get_cleaning_gates
from src.utils.sandbox_deps import (
    BASE_ALLOWLIST,
    EXTENDED_ALLOWLIST,
    CLOUDRUN_NATIVE_ALLOWLIST,
    CLOUDRUN_OPTIONAL_ALLOWLIST,
    BANNED_ALWAYS_ALLOWLIST,
)
from src.utils.llm_fallback import call_chat_with_fallback, extract_response_text
from openai import OpenAI

load_dotenv()


class DataEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Data Engineer Agent with OpenRouter primary + fallback.
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API Key is required.")

        timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS")
        try:
            timeout_seconds = float(timeout_raw) if timeout_raw else 120.0
        except ValueError:
            timeout_seconds = 120.0
        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_X_TITLE")
        if title:
            headers["X-Title"] = title
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": timeout_seconds,
        }
        if headers:
            client_kwargs["default_headers"] = headers
        self.client = OpenAI(**client_kwargs)

        self.model_name = (
            os.getenv("DATA_ENGINEER_PRIMARY_MODEL")
            or os.getenv("OPENROUTER_DE_PRIMARY_MODEL")
            or os.getenv("DEEPSEEK_DE_PRIMARY_MODEL")
            or "minimax/minimax-m2.5"
        )
        self.fallback_model_name = (
            os.getenv("DATA_ENGINEER_FALLBACK_MODEL")
            or os.getenv("OPENROUTER_DE_FALLBACK_MODEL")
            or "moonshotai/kimi-k2.5"
        )

        self.last_prompt = None
        self.last_response = None

    def _extract_nonempty(self, response) -> str:
        """
        Extracts non-empty content from LLM response.
        Raises ValueError("EMPTY_COMPLETION") if content is empty (CAUSA RAIZ 2).
        This triggers retry logic in call_with_retries.
        """
        content = extract_response_text(response)

        if not content:
            print("ERROR: LLM returned EMPTY_COMPLETION. Will retry.")
            raise ValueError("EMPTY_COMPLETION")

        return content

    def _build_runtime_dependency_context(self) -> Dict[str, Any]:
        """
        Build a compact runtime dependency contract for DE prompts.
        This gives the model explicit import allowlist + version hints
        to reduce runtime incompatibilities across sandbox images.
        """
        requirements_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "cloudrun",
                "heavy_runner",
                "requirements.txt",
            )
        )
        pinned_specs: Dict[str, str] = {}
        try:
            if os.path.exists(requirements_path):
                with open(requirements_path, "r", encoding="utf-8") as f_req:
                    for raw in f_req:
                        line = str(raw or "").strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("-"):
                            continue
                        token = line.split(";", 1)[0].strip()
                        token = token.split("#", 1)[0].strip()
                        if not token:
                            continue
                        for op in ("==", ">=", "<=", "~=", "!=", ">", "<"):
                            if op in token:
                                name, spec = token.split(op, 1)
                                pkg = name.strip().lower()
                                if pkg:
                                    pinned_specs[pkg] = f"{op}{spec.strip()}"
                                break
        except Exception:
            pinned_specs = {}

        pandas_spec = pinned_specs.get("pandas", "unbounded")

        return {
            "backend_profile": "cloudrun",
            "allowlist": {
                "base": sorted({str(item) for item in BASE_ALLOWLIST if str(item).strip()}),
                "extended_optional": sorted(
                    {str(item) for item in EXTENDED_ALLOWLIST if str(item).strip()}
                ),
                "cloudrun_native": sorted(
                    {str(item) for item in CLOUDRUN_NATIVE_ALLOWLIST if str(item).strip()}
                ),
                "cloudrun_optional": sorted(
                    {str(item) for item in CLOUDRUN_OPTIONAL_ALLOWLIST if str(item).strip()}
                ),
            },
            "blocked_always": sorted(
                {str(item) for item in BANNED_ALWAYS_ALLOWLIST if str(item).strip()}
            ),
            "version_hints": {
                "python": "3.11",
                "pandas": pandas_spec,
            },
            "guidance": [
                "Import only allowlisted roots.",
                "Use stable public APIs compatible with version_hints.",
                "Avoid deprecated kwargs/behaviors when equivalent safe idioms exist.",
            ],
        }

    def _build_contract_focus_context(
        self,
        contract: Dict[str, Any],
        de_view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a compact DE-focused contract context to reduce prompt noise.
        """
        contract = contract if isinstance(contract, dict) else {}
        de_view = de_view if isinstance(de_view, dict) else {}
        focus: Dict[str, Any] = {}

        for key in ("scope", "required_outputs", "column_roles", "outlier_policy"):
            value = contract.get(key)
            if value not in (None, "", [], {}):
                focus[key] = value

        artifact_requirements = contract.get("artifact_requirements")
        if isinstance(artifact_requirements, dict):
            clean_dataset = artifact_requirements.get("clean_dataset")
            if isinstance(clean_dataset, dict) and clean_dataset:
                focus["artifact_requirements"] = {"clean_dataset": clean_dataset}

        if de_view:
            view_focus: Dict[str, Any] = {}
            for key in (
                "required_columns",
                "required_feature_selectors",
                "optional_passthrough_columns",
                "output_path",
                "output_manifest_path",
                "cleaning_gates",
                "column_transformations",
            ):
                value = de_view.get(key)
                if value not in (None, "", [], {}):
                    view_focus[key] = value
            if view_focus:
                focus["de_view_projection"] = view_focus

        return focus

    def _detect_repair_mode(self, data_audit: str, requested: bool) -> bool:
        """
        Detect if DE should operate in repair mode.
        """
        if requested:
            return True
        text = str(data_audit or "")
        repair_markers = (
            "RUNTIME_ERROR_CONTEXT",
            "PREFLIGHT_ERROR_CONTEXT",
            "STATIC_SCAN_VIOLATIONS",
            "UNDEFINED_NAME_GUARD",
            "CLEANING_REVIEWER_ALERT",
            "LLM_FAILURE_EXPLANATION",
        )
        return any(marker in text for marker in repair_markers)

    def generate_cleaning_script(
        self,
        data_audit: str,
        strategy: Dict[str, Any],
        input_path: str,
        business_objective: str = "",
        csv_encoding: str = "utf-8",
        csv_sep: str = ",",
        csv_decimal: str = ".",
        execution_contract: Optional[Dict[str, Any]] = None,
        contract_min: Optional[Dict[str, Any]] = None,
        de_view: Optional[Dict[str, Any]] = None,
        repair_mode: bool = False,
    ) -> str:
        """
        Generates a Python script to clean and standardize the dataset.
        """
        from src.utils.prompting import render_prompt

        contract = execution_contract or contract_min or {}
        from src.utils.context_pack import compress_long_lists, summarize_long_list, COLUMN_LIST_POINTER

        de_view = de_view or {}
        contract_context = contract if isinstance(contract, dict) else {}
        if not contract_context:
            de_contract_context = {}
            for key in (
                "required_columns",
                "optional_passthrough_columns",
                "output_path",
                "output_manifest_path",
                "manifest_path",
                "output_dialect",
                "cleaning_gates",
                "data_engineer_runbook",
                "constraints",
            ):
                value = de_view.get(key)
                if value in (None, "", [], {}):
                    continue
                if key == "manifest_path" and "output_manifest_path" in de_contract_context:
                    continue
                de_contract_context[key] = value
            contract_context = de_contract_context
        contract_context = self._build_contract_focus_context(contract_context, de_view or {})
        contract_json = json.dumps(compress_long_lists(contract_context)[0], indent=2)
        de_view_json = json.dumps(compress_long_lists(de_view)[0], indent=2)
        de_output_path = str(de_view.get("output_path") or "").strip()
        de_manifest_path = str(
            de_view.get("output_manifest_path")
            or de_view.get("manifest_path")
            or ""
        ).strip()
        view_cleaning_gates = get_cleaning_gates({"cleaning_gates": de_view.get("cleaning_gates")})
        cleaning_gates = (
            get_cleaning_gates(contract)
            or get_cleaning_gates(execution_contract or {})
            or view_cleaning_gates
            or []
        )
        cleaning_gates_json = json.dumps(compress_long_lists(cleaning_gates)[0], indent=2)

        # V4.1: Use data_engineer_runbook only, no legacy role_runbooks fallback
        de_runbook = contract.get("data_engineer_runbook")
        if de_runbook in (None, "", [], {}):
            de_runbook = de_view.get("data_engineer_runbook")
        if isinstance(de_runbook, str):
            de_runbook = de_runbook.strip()
        if de_runbook in (None, "", [], {}):
            de_runbook = {}
        de_runbook_json = json.dumps(compress_long_lists(de_runbook)[0], indent=2)
        outlier_policy = de_view.get("outlier_policy")
        if not isinstance(outlier_policy, dict) or not outlier_policy:
            policy_from_contract = contract.get("outlier_policy")
            outlier_policy = policy_from_contract if isinstance(policy_from_contract, dict) else {}
        if not isinstance(outlier_policy, dict):
            outlier_policy = {}
        outlier_policy_json = json.dumps(compress_long_lists(outlier_policy)[0], indent=2)
        outlier_report_path = str(
            de_view.get("outlier_report_path")
            or outlier_policy.get("report_path")
            or ""
        ).strip()
        if not outlier_report_path and outlier_policy:
            outlier_enabled = outlier_policy.get("enabled")
            if isinstance(outlier_enabled, str):
                outlier_enabled = outlier_enabled.strip().lower() in {"1", "true", "yes", "on", "enabled"}
            if outlier_enabled is None:
                outlier_enabled = bool(
                    outlier_policy.get("target_columns")
                    or outlier_policy.get("methods")
                    or outlier_policy.get("treatment")
                )
            outlier_stage = str(outlier_policy.get("apply_stage") or "data_engineer").strip().lower()
            if bool(outlier_enabled) and outlier_stage in {"data_engineer", "both"}:
                outlier_report_path = "data/outlier_treatment_report.json"
        artifact_requirements = contract.get("artifact_requirements")
        clean_dataset_cfg = {}
        if isinstance(artifact_requirements, dict):
            clean_dataset_candidate = artifact_requirements.get("clean_dataset")
            if isinstance(clean_dataset_candidate, dict):
                clean_dataset_cfg = clean_dataset_candidate
        column_transformations = de_view.get("column_transformations")
        if not isinstance(column_transformations, dict) or not column_transformations:
            column_transformations = clean_dataset_cfg.get("column_transformations")
        if not isinstance(column_transformations, dict):
            column_transformations = {}
        column_transformations_json = json.dumps(
            compress_long_lists(column_transformations)[0],
            indent=2,
        )
        runtime_dependency_context = self._build_runtime_dependency_context()
        runtime_dependency_context_json = json.dumps(
            compress_long_lists(runtime_dependency_context)[0], indent=2
        )
        effective_repair_mode = self._detect_repair_mode(data_audit, repair_mode)
        operating_mode = "REPAIR" if effective_repair_mode else "INITIAL"
        repair_notes = (
            "Patch root causes from error context first. Keep working logic stable."
            if effective_repair_mode
            else "Build the first executable script from contract and context."
        )

        # [SAFETY] Truncate data_audit if massive to prevent context overflow
        # The audit concatenates many sources; preserve head (structure) and tail (recent instructions).
        if data_audit and len(data_audit) > 100000:
            print(f"DEBUG: Truncating massive data_audit ({len(data_audit)} chars) to 100k.")
            head_len = 50000
            tail_len = 50000
            data_audit = data_audit[:head_len] + "\n...[AUDIT TRUNCATED FOR CONTEXT SAFETY]...\n" + data_audit[-tail_len:]
        SYSTEM_TEMPLATE = """
        You are a Senior Data Engineer producing a deterministic cleaning script.

        === SENIOR ENGINEERING PROTOCOL ===
        $senior_engineering_protocol

        OPERATING_MODE: $operating_mode
        MODE_NOTE: $repair_notes

        SOURCE-OF-TRUTH PRECEDENCE:
        1) CLEANING_GATES_CONTEXT + required_columns + required_feature_selectors from DE_VIEW_CONTEXT
        2) COLUMN_TRANSFORMATIONS_CONTEXT
        3) ROLE RUNBOOK (advisory)

        CORE CONSTRAINTS:
        - Output valid Python code only (no markdown/code fences, no prose, no JSON plans).
        - No network/system calls (requests/subprocess/os.system forbidden).
        - No pandas private APIs.
        - Use only dependencies allowed by RUNTIME_DEPENDENCY_CONTEXT.
        - Script must parse as valid Python.

        SCOPE:
        - Cleaning only (no modeling/scoring/optimization/analytics).
        - Read input with pd.read_csv(..., dtype=str, low_memory=False, sep, decimal, encoding from inputs).
        - Write cleaned CSV to $de_output_path.
        - Write manifest to $de_manifest_path.
        - If outlier policy is enabled for data_engineer/both, write report to $outlier_report_path.

        REQUIRED MANIFEST CONTENT:
        - output_dialect
        - row_counts
        - conversions
        - contract_conflicts_resolved (empty list if none)

        COLUMN BEHAVIOR:
        - required_columns are anchor columns.
        - Expand required_feature_selectors against input header; expanded columns are required unless
          explicitly dropped by COLUMN_TRANSFORMATIONS_CONTEXT + allowed drop_policy evidence.
        - Keep optional passthrough columns only if listed and present.
        - Missing required columns -> fail fast with ValueError.
        - Never fabricate columns.

        TARGET/PARTITION SAFETY:
        - Do not impute outcome/target columns unless contract explicitly requests it.
        - Preserve missingness for partially labeled targets.
        - Preserve split/partition columns when present.
        - If integer target can include nulls, use pandas nullable integer dtype.

        REPAIR RULES (when OPERATING_MODE is REPAIR):
        - Use RUNTIME_ERROR_CONTEXT/PREFLIGHT_ERROR_CONTEXT to fix root causes first.
        - Keep already-correct logic stable; avoid unrelated rewrites.
        - Prioritize executable syntax and required artifacts.

        TOP-OF-SCRIPT COMMENT BLOCKS (required):
        # Decision Log:
        # Assumptions:
        # Risks & Checks:

        INPUT PARAMETERS:
        - Input: '$input_path'
        - Encoding: '$csv_encoding' | Sep: '$csv_sep' | Decimal: '$csv_decimal'
        - DE Cleaning Objective: "$business_objective"
        - Required Columns (DE View): $required_columns
        - Optional Passthrough Columns (keep if present): $optional_passthrough_columns
        - DE_VIEW_CONTEXT (json): $de_view_context
        - OUTLIER_POLICY_CONTEXT (json): $outlier_policy_context
        - COLUMN_TRANSFORMATIONS_CONTEXT (json): $column_transformations_context
        - EXECUTION_CONTRACT_CONTEXT (json): $execution_contract_context
        - CLEANING_GATES_CONTEXT (json): $cleaning_gates_context
        - RUNTIME_DEPENDENCY_CONTEXT (json): $runtime_dependency_context
        - ROLE RUNBOOK (Data Engineer): $data_engineer_runbook

        DATA AUDIT:
        $data_audit

        CONTRACT-CHECKLIST BEFORE RETURNING (silent self-check):
        - Python parses without SyntaxError.
        - Required output paths are used.
        - HARD gates fail with ValueError("CLEANING_GATE_FAILED: ...") when violated.
        - No markdown/code fences in output.
        """

        USER_TEMPLATE = "Generate the cleaning script following Principles."

        # Rendering
        required_columns_payload = de_view.get("required_columns") or strategy.get("required_columns", [])
        if isinstance(required_columns_payload, list) and len(required_columns_payload) > 80:
            required_columns_payload = summarize_long_list(required_columns_payload)
            required_columns_payload["note"] = COLUMN_LIST_POINTER
        optional_passthrough_payload = de_view.get("optional_passthrough_columns") or []
        if isinstance(optional_passthrough_payload, list) and len(optional_passthrough_payload) > 80:
            optional_passthrough_payload = summarize_long_list(optional_passthrough_payload)
            optional_passthrough_payload["note"] = COLUMN_LIST_POINTER

        system_prompt = render_prompt(
            SYSTEM_TEMPLATE,
            input_path=input_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            business_objective=business_objective,
            required_columns=json.dumps(required_columns_payload),
            optional_passthrough_columns=json.dumps(optional_passthrough_payload),
            data_audit=data_audit,
            execution_contract_context=contract_json,
            de_view_context=de_view_json,
            outlier_policy_context=outlier_policy_json,
            column_transformations_context=column_transformations_json,
            data_engineer_runbook=de_runbook_json,
            cleaning_gates_context=cleaning_gates_json,
            runtime_dependency_context=runtime_dependency_context_json,
            senior_engineering_protocol=SENIOR_ENGINEERING_PROTOCOL,
            de_output_path=de_output_path,
            de_manifest_path=de_manifest_path,
            outlier_report_path=outlier_report_path,
            operating_mode=operating_mode,
            repair_notes=repair_notes,
        )
        self.last_prompt = system_prompt + "\n\nUSER:\n" + USER_TEMPLATE
        print(f"DEBUG: DE System Prompt Len: {len(system_prompt)}")
        print(f"DEBUG: DE System Prompt Preview: {system_prompt[:300]}...")
        if len(system_prompt) < 100:
            print("CRITICAL: System Prompt is suspiciously short!")

        from src.utils.retries import call_with_retries

        def _call_model():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TEMPLATE},
            ]
            response, model_used = call_chat_with_fallback(
                self.client,
                messages,
                [self.model_name, self.fallback_model_name],
                call_kwargs={"temperature": 0.1},
                logger=self.logger,
                context_tag="data_engineer",
            )
            self.logger.info("DATA_ENGINEER_MODEL_USED: %s", model_used)
            # Use _extract_nonempty to handle EMPTY_COMPLETION (CAUSA RA?Z 2)
            content = self._extract_nonempty(response)
            print(f"DEBUG: Primary DE Response Preview: {content[:200]}...")
            self.last_response = content

            # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("LLM Server Timeout (504 Received)")

            # Check for JSON error messages that are NOT valid code
            content_stripped = content.strip()
            if content_stripped.startswith("{") or content_stripped.startswith("["):
                try:
                    import json
                    json_content = json.loads(content_stripped)
                    if isinstance(json_content, dict):
                        if "error" in json_content or "errorMessage" in json_content:
                            raise ConnectionError(f"API Error Detected (JSON): {content_stripped}")
                except Exception:
                    pass

            # Text based fallback for Error/Overloaded keywords
            content_lower = content.lower()
            if "error" in content_lower and ("overloaded" in content_lower or "rate limit" in content_lower or "429" in content_lower):
                raise ConnectionError(f"API Error Detected (Text): {content_stripped}")

            return content

        injection = "\n".join(
            [
                "import os",
                "import json",
                "from datetime import date, datetime",
                "try:",
                "    import numpy as np",
                "except Exception:",
                "    np = None",
                "try:",
                "    import pandas as pd",
                "except Exception:",
                "    pd = None",
                "",
                "os.makedirs('data', exist_ok=True)",
                "",
                "def _to_jsonable(value):",
                "    if value is None:",
                "        return None",
                "    if isinstance(value, (str, int, bool)):",
                "        return value",
                "    if isinstance(value, float):",
                "        return None if value != value else value",
                "    if isinstance(value, (datetime, date)):",
                "        return value.isoformat()",
                "    if isinstance(value, (list, tuple, set)):",
                "        return [_to_jsonable(item) for item in value]",
                "    if isinstance(value, dict):",
                "        return {str(k): _to_jsonable(v) for k, v in value.items()}",
                "    if isinstance(value, (bytes, bytearray)):",
                "        return value.decode('utf-8', errors='replace')",
                "    if np is not None:",
                "        if isinstance(value, np.bool_):",
                "            return bool(value)",
                "        if isinstance(value, np.integer):",
                "            return int(value)",
                "        if isinstance(value, np.floating):",
                "            return float(value)",
                "        if isinstance(value, np.ndarray):",
                "            return [_to_jsonable(item) for item in value.tolist()]",
                "    if pd is not None:",
                "        if value is pd.NA:",
                "            return None",
                "        if isinstance(value, pd.Timestamp):",
                "            return value.isoformat()",
                "        try:",
                "            if pd.isna(value) is True:",
                "                return None",
                "        except Exception:",
                "            pass",
                "    return str(value)",
                "",
                "_ORIG_JSON_DUMP = json.dump",
                "_ORIG_JSON_DUMPS = json.dumps",
                "",
                "def _safe_dump_json(obj, fp, **kwargs):",
                "    payload = _to_jsonable(obj)",
                "    kwargs.pop('default', None)",
                "    return _ORIG_JSON_DUMP(payload, fp, **kwargs)",
                "",
                "def _safe_dumps_json(obj, **kwargs):",
                "    payload = _to_jsonable(obj)",
                "    kwargs.pop('default', None)",
                "    return _ORIG_JSON_DUMPS(payload, **kwargs)",
                "",
                "json.dump = _safe_dump_json",
                "json.dumps = _safe_dumps_json",
                "",
            ]
        ) + "\n"

        try:
            content = call_with_retries(_call_model, max_retries=5, backoff_factor=2, initial_delay=2)
            print("DEBUG: OpenRouter response received.")

            code = self._clean_code(content)

            return injection + code

        except Exception as e:
            error_msg = f"Data Engineer Failed (Primary & Fallback): {str(e)}"
            print(f"CRITICAL: {error_msg}")
            return f"# Error: {error_msg}"

    def _clean_code(self, code: str) -> str:
        """
        Extracts code from markdown blocks, validates syntax, and applies auto-fixes.
        Raises ValueError if code is empty or has unfixable syntax errors (CAUSA RAIZ 2 & 3).
        """
        cleaned = (extract_code_block(code) or "").strip()
        if not cleaned and not (code or "").strip():
            print("ERROR: EMPTY_CODE_AFTER_EXTRACTION")
            raise ValueError("EMPTY_CODE_AFTER_EXTRACTION")

        def _autofix_assign_digit_identifier(src: str) -> str:
            # .assign(1stYearAmount=...) -> .assign(**{'1stYearAmount': ...})
            pattern = r'\.assign\(\s*([0-9][a-zA-Z0-9_]*)\s*=\s*([^)]+)\)'

            def fix_assign(match):
                col_name = match.group(1)
                value = match.group(2)
                return f".assign(**{{'{col_name}': {value}}})"

            return re.sub(pattern, fix_assign, src or "")

        def _trim_to_code_start(src: str) -> str:
            if not isinstance(src, str):
                return ""
            normalized = re.sub(r"</?think>", "\n", src, flags=re.IGNORECASE).strip()
            if not normalized:
                return ""
            lines = normalized.splitlines()
            start_pattern = re.compile(
                r"^(#|from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|if\s+__name__|if\s+|for\s+|while\s+|try:|with\s+|@|[A-Za-z_]\w*\s*=|print\()"
            )
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if start_pattern.match(stripped):
                    return "\n".join(lines[idx:]).strip()
            return normalized

        def _looks_like_script(src: str) -> bool:
            if not isinstance(src, str):
                return False
            text = src.strip()
            if not text:
                return False
            code_pattern = re.compile(
                r"(?m)^\s*(from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|if\s+__name__|"
                r"if\s+|for\s+|while\s+|try:|with\s+|@\w+|[A-Za-z_]\w*\s*=|print\(|raise\s+)"
            )
            return bool(code_pattern.search(text))

        candidates: List[str] = []

        def _push_candidate(value: str) -> None:
            if not isinstance(value, str):
                return
            stripped = value.strip()
            if not stripped or stripped in candidates:
                return
            candidates.append(stripped)

        _push_candidate(cleaned)
        _push_candidate(code or "")

        raw = code or ""
        if "```" in raw:
            parts = [p.strip() for p in re.split(r"```(?:python)?", raw, flags=re.IGNORECASE) if isinstance(p, str)]
            for part in parts:
                _push_candidate(part)
            first_fence = re.search(r"```(?:python)?", raw, re.IGNORECASE)
            if first_fence:
                _push_candidate(raw[: first_fence.start()])
                _push_candidate(raw[first_fence.end() :])

        think_tail = re.split(r"</think>", raw, flags=re.IGNORECASE)
        if len(think_tail) > 1:
            _push_candidate(think_tail[-1])

        recovery_candidates: List[str] = []
        for candidate in candidates:
            recovery_candidates.append(candidate)
            trimmed = _trim_to_code_start(candidate)
            if trimmed and trimmed != candidate:
                recovery_candidates.append(trimmed)

        last_syntax_error: Optional[SyntaxError] = None
        for candidate in recovery_candidates:
            for variant in (candidate, _autofix_assign_digit_identifier(candidate)):
                variant = (variant or "").strip()
                if not variant or not _looks_like_script(variant):
                    continue
                try:
                    ast.parse(variant)
                    if variant != candidate:
                        print("DEBUG: Auto-fix successful.")
                    return variant
                except SyntaxError as e:
                    last_syntax_error = e

        if last_syntax_error:
            print(f"ERROR: Auto-fix failed. Syntax still invalid: {last_syntax_error}")
            raise ValueError(f"INVALID_PYTHON_SYNTAX: {last_syntax_error}")
        print("ERROR: EMPTY_CODE_AFTER_EXTRACTION")
        raise ValueError("EMPTY_CODE_AFTER_EXTRACTION")

