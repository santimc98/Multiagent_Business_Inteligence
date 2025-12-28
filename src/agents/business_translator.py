import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from string import Template
import json
from typing import Dict, Any, Optional, List
import csv
from src.utils.prompting import render_prompt


def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _safe_load_csv(path: str, max_rows: int = 200):
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for _, row in zip(range(max_rows), reader):
                rows.append(row)
            return {
                "columns": reader.fieldnames or [],
                "rows": rows,
                "row_count_sampled": len(rows),
            }
    except Exception:
        return None

def _summarize_numeric_columns(rows: List[Dict[str, Any]], columns: List[str], max_cols: int = 12):
    numeric_summary = {}
    for col in columns:
        values = []
        for row in rows:
            raw = row.get(col)
            if raw is None or raw == "":
                continue
            try:
                values.append(float(str(raw).replace(",", ".")))
            except Exception:
                continue
        if values:
            numeric_summary[col] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "n": len(values),
            }
        if len(numeric_summary) >= max_cols:
            break
    return numeric_summary

def _pick_top_examples(rows: List[Dict[str, Any]], columns: List[str], value_keys: List[str], label_keys: List[str], max_rows: int = 3):
    if not rows or not columns:
        return None
    value_key = None
    for key in value_keys:
        if key in columns:
            value_key = key
            break
    if not value_key:
        return None
    def _coerce_num(row):
        raw = row.get(value_key)
        if raw is None or raw == "":
            return None
        try:
            return float(str(raw).replace(",", "."))
        except Exception:
            return None
    scored = []
    for row in rows:
        val = _coerce_num(row)
        if val is None:
            continue
        scored.append((val, row))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    examples = []
    for val, row in scored[:max_rows]:
        example = {"value_key": value_key, "value": val}
        for label in label_keys:
            if label in row and row.get(label) not in (None, ""):
                example[label] = row.get(label)
        examples.append(example)
    return examples

def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False

def _extract_numeric_metrics(metrics: Dict[str, Any], max_items: int = 8):
    if not isinstance(metrics, dict):
        return []
    items = []
    for key, value in metrics.items():
        if _is_number(value):
            items.append((str(key), float(value)))
        if len(items) >= max_items:
            break
    return items

def _build_fact_cards(case_summary_ctx, scored_rows_ctx, weights_ctx, data_adequacy_ctx, max_items: int = 8):
    facts = []
    if isinstance(case_summary_ctx, dict):
        examples = case_summary_ctx.get("examples") or []
        for example in examples:
            value_key = example.get("value_key")
            value = example.get("value")
            if value_key and _is_number(value):
                labels = {k: v for k, v in example.items() if k not in {"value_key", "value"}}
                facts.append({
                    "source": "case_summary.csv",
                    "metric": value_key,
                    "value": float(value),
                    "labels": labels,
                })
    if isinstance(scored_rows_ctx, dict):
        examples = scored_rows_ctx.get("examples") or []
        for example in examples:
            value_key = example.get("value_key")
            value = example.get("value")
            if value_key and _is_number(value):
                labels = {k: v for k, v in example.items() if k not in {"value_key", "value"}}
                facts.append({
                    "source": "scored_rows.csv",
                    "metric": value_key,
                    "value": float(value),
                    "labels": labels,
                })
    if isinstance(weights_ctx, dict):
        for key in ("metrics", "classification", "regression", "propensity_model", "price_model"):
            metrics = weights_ctx.get(key)
            for metric_key, metric_val in _extract_numeric_metrics(metrics):
                facts.append({
                    "source": "weights.json",
                    "metric": metric_key,
                    "value": metric_val,
                    "labels": {"model_block": key},
                })
    if isinstance(data_adequacy_ctx, dict):
        signals = data_adequacy_ctx.get("signals", {})
        for metric_key, metric_val in _extract_numeric_metrics(signals):
            facts.append({
                "source": "data_adequacy_report.json",
                "metric": metric_key,
                "value": metric_val,
                "labels": {},
            })
    return facts[:max_items]

class BusinessTranslatorAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Business Translator Agent with Gemini 3 Flash.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            generation_config={"temperature": 0.7},
        )

    def generate_report(self, state: Dict[str, Any], error_message: Optional[str] = None, has_partial_visuals: bool = False, plots: Optional[List[str]] = None) -> str:
        
        # Sanitize Visuals Context
        has_partial_visuals = bool(has_partial_visuals)
        plots = plots or []
        artifact_index = state.get("artifact_index") or []
        artifact_index = [a for a in artifact_index if isinstance(a, str)]

        def _artifact_available(path: str) -> bool:
            if artifact_index:
                return path in artifact_index
            return os.path.exists(path)
        
        # Safe extraction of strategy info
        strategy = state.get('selected_strategy', {})
        strategy_title = strategy.get('title', 'General Analysis')
        hypothesis = strategy.get('hypothesis', 'N/A')
        analysis_type = str(strategy.get('analysis_type', 'N/A'))
        
        # Review content
        review_verdict = state.get("review_verdict")
        if review_verdict:
            compliance = review_verdict
        else:
            review = state.get('review_feedback', {})
            if isinstance(review, dict):
                compliance = review.get('status', 'PENDING')
            else:
                # If it's a string (e.g. just the feedback text from older legacy flows or simple strings)
                compliance = "REVIEWED" if review else "PENDING"
        
        # Construct JSON Context for Visuals safely using json library
        visuals_context_data = {
            "has_partial_visuals": has_partial_visuals,
            "plots_count": len(plots),
            "plots_list": plots
        }
        visuals_context_json = json.dumps(visuals_context_data, ensure_ascii=False)
        
        # Load optional artifacts for context
        contract = _safe_load_json("data/execution_contract.json") or {}
        integrity_audit = _safe_load_json("data/integrity_audit_report.json") or {}
        output_contract_report = _safe_load_json("data/output_contract_report.json") or {}
        case_alignment_report = _safe_load_json("data/case_alignment_report.json") or {}
        data_adequacy_report = _safe_load_json("data/data_adequacy_report.json") or {}
        plot_insights = _safe_load_json("data/plot_insights.json") or {}
        steward_summary = _safe_load_json("data/steward_summary.json") or {}
        cleaning_manifest = _safe_load_json("data/cleaning_manifest.json") or {}
        run_summary = _safe_load_json("data/run_summary.json") or {}
        weights_payload = _safe_load_json("data/weights.json") if _artifact_available("data/weights.json") else None
        weights_payload = weights_payload or {}
        scored_rows = _safe_load_csv("data/scored_rows.csv") if _artifact_available("data/scored_rows.csv") else None
        case_summary = _safe_load_csv("data/case_summary.csv") if _artifact_available("data/case_summary.csv") else None
        cleaned_rows = _safe_load_csv("data/cleaned_data.csv", max_rows=100) if _artifact_available("data/cleaned_data.csv") else None
        business_objective = state.get("business_objective") or contract.get("business_objective") or ""

        def _summarize_integrity():
            issues = integrity_audit.get("issues", []) if isinstance(integrity_audit, dict) else []
            severity_counts = {}
            for i in issues:
                sev = str(i.get("severity", "unknown"))
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            top = issues[:3]
            return f"Issues by severity: {severity_counts}; Top3: {top}"

        def _summarize_contract():
            if not contract:
                return "No execution contract."
            return {
                "strategy_title": contract.get("strategy_title"),
                "business_objective": contract.get("business_objective"),
                "required_outputs": contract.get("required_outputs", []),
                "validations": contract.get("validations", []),
                "quality_gates": contract.get("quality_gates", {}),
                "business_alignment": contract.get("business_alignment", {}),
                "spec_extraction": contract.get("spec_extraction", {}),
                "iteration_policy": contract.get("iteration_policy", {}),
            }

        def _summarize_output_contract():
            if not output_contract_report:
                return "No output contract report."
            miss = output_contract_report.get("missing", [])
            present = output_contract_report.get("present", [])
            return f"Outputs present={len(present)} missing={len(miss)}"

        def _summarize_steward():
            if not steward_summary:
                return "No steward summary."
            summary = steward_summary.get("summary", "")
            encoding = steward_summary.get("encoding")
            sep = steward_summary.get("sep")
            decimal = steward_summary.get("decimal")
            return {
                "summary": summary[:1200],
                "encoding": encoding,
                "sep": sep,
                "decimal": decimal,
                "file_size_bytes": steward_summary.get("file_size_bytes"),
            }

        def _summarize_cleaning():
            if not cleaning_manifest:
                return "No cleaning manifest."
            row_counts = cleaning_manifest.get("row_counts", {})
            conversions = cleaning_manifest.get("conversions", {})
            dropped = cleaning_manifest.get("dropped_rows", {})
            conversion_keys = []
            if isinstance(conversions, dict):
                conversion_keys = list(conversions.keys())[:12]
            elif isinstance(conversions, list):
                conversion_keys = [c.get("column") for c in conversions if isinstance(c, dict) and c.get("column")]
                conversion_keys = conversion_keys[:12]
            return {
                "row_counts": row_counts,
                "dropped_rows": dropped,
                "conversion_keys": conversion_keys,
            }

        def _summarize_weights():
            if not weights_payload:
                return "No weights/metrics payload."
            if not isinstance(weights_payload, dict):
                return weights_payload
            summary = {}
            for key in ("metrics", "weights", "propensity_model", "price_model", "optimization", "regression", "classification"):
                if key in weights_payload:
                    summary[key] = weights_payload.get(key)
            if not summary:
                summary["keys"] = list(weights_payload.keys())[:12]
            return summary

        def _summarize_model_metrics():
            metrics_summary = {}
            if isinstance(weights_payload, dict):
                for key in ("metrics", "propensity_model", "price_model", "optimization", "regression", "classification"):
                    if key in weights_payload:
                        metrics_summary[key] = weights_payload.get(key)
            if isinstance(run_summary, dict):
                run_metrics = run_summary.get("metrics")
                if run_metrics:
                    metrics_summary["run_summary_metrics"] = run_metrics
            if not metrics_summary:
                return "No explicit model metrics found."
            return metrics_summary

        def _summarize_case_summary():
            if not case_summary:
                return "No case_summary.csv."
            columns = case_summary.get("columns", [])
            rows = case_summary.get("rows", [])
            if "metric" in columns and "value" in columns:
                metrics = {}
                for row in rows:
                    key = row.get("metric")
                    if not key:
                        continue
                    raw_value = row.get("value")
                    if raw_value is None or raw_value == "":
                        metrics[key] = None
                        continue
                    try:
                        metrics[key] = float(str(raw_value).replace(",", "."))
                    except Exception:
                        metrics[key] = raw_value
                return {
                    "row_count_sampled": case_summary.get("row_count_sampled", 0),
                    "columns": columns,
                    "metrics": metrics,
                }
            numeric_summary = _summarize_numeric_columns(rows, columns)
            examples = _pick_top_examples(
                rows,
                columns,
                value_keys=[
                    "Expected_Value_mean",
                    "ExpectedValue_mean",
                    "Expected_Value",
                    "ExpectedValue",
                    "Predicted_Price_mean",
                    "Predicted_Price",
                ],
                label_keys=["Sector", "Size_Decile", "Debtors_Decile", "Segment", "Typology_Cluster"],
            )
            return {
                "row_count_sampled": case_summary.get("row_count_sampled", 0),
                "columns": columns,
                "numeric_summary": numeric_summary,
                "examples": examples,
            }

        def _summarize_scored_rows():
            if not scored_rows:
                return "No scored_rows.csv."
            columns = scored_rows.get("columns", [])
            rows = scored_rows.get("rows", [])
            numeric_summary = _summarize_numeric_columns(rows, columns)
            examples = _pick_top_examples(
                rows,
                columns,
                value_keys=[
                    "Expected_Value",
                    "ExpectedValue",
                    "expected_revenue_pred",
                    "ExpectedRevenue",
                ],
                label_keys=["Sector", "Account", "FiscalId", "Size", "Debtors"],
            )
            return {
                "row_count_sampled": scored_rows.get("row_count_sampled", 0),
                "columns": columns,
                "numeric_summary": numeric_summary,
                "examples": examples,
            }

        def _summarize_run():
            if not run_summary:
                return "No run_summary.json."
            return {
                "status": run_summary.get("status"),
                "failed_gates": run_summary.get("failed_gates", []),
                "warnings": run_summary.get("warnings", []),
                "metrics": run_summary.get("metrics", {}),
            }

        def _summarize_gate_context():
            gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context") or {}
            if not gate_context:
                return "No gate context."
            if isinstance(gate_context, dict):
                return {
                    "source": gate_context.get("source"),
                    "status": gate_context.get("status"),
                    "failed_gates": gate_context.get("failed_gates", []),
                    "required_fixes": gate_context.get("required_fixes", []),
                }
            return str(gate_context)[:1200]

        def _summarize_review_feedback():
            feedback = state.get("review_feedback") or state.get("execution_feedback") or ""
            if isinstance(feedback, dict):
                return feedback
            if isinstance(feedback, str):
                return feedback[:2000]
            return str(feedback)[:2000]

        def _summarize_data_adequacy():
            if not data_adequacy_report:
                return "No data adequacy report."
            return {
                "status": data_adequacy_report.get("status"),
                "reasons": data_adequacy_report.get("reasons", []),
                "recommendations": data_adequacy_report.get("recommendations", []),
                "signals": data_adequacy_report.get("signals", {}),
                "quality_gates_alignment": data_adequacy_report.get("quality_gates_alignment", {}),
                "consecutive_data_limited": data_adequacy_report.get("consecutive_data_limited"),
                "data_limited_threshold": data_adequacy_report.get("data_limited_threshold"),
                "threshold_reached": data_adequacy_report.get("threshold_reached"),
            }

        def _summarize_case_alignment():
            if not case_alignment_report:
                return "No case alignment report."
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            return f"Status={status}; Failures={failures}; KeyMetrics={metrics}"

        def _case_alignment_business_status():
            if not case_alignment_report:
                return {
                    "label": "NO_DATA",
                    "status": "UNKNOWN",
                    "message": "No se encontró reporte de alineación de casos.",
                    "recommendation": "Revisar si el proceso generó data/case_alignment_report.json.",
                }
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            if status == "SKIPPED":
                return {
                    "label": "PENDIENTE_DEFINICION_GATES",
                    "status": "SKIPPED",
                    "message": "No se definieron gates de alineación de casos en el contrato.",
                    "recommendation": "Definir métricas y umbrales en el contrato para evaluar preparación de negocio.",
                }
            if status == "PASS":
                return {
                    "label": "APTO_CONDICIONAL",
                    "status": "PASS",
                    "message": "La alineación con la lógica de casos cumple los umbrales definidos.",
                    "key_metrics": metrics,
                }
            # FAIL
            details = []
            for failure in failures:
                metric_val = metrics.get(failure)
                thresh_val = thresholds.get(failure.replace("case_means", "min"), thresholds.get(failure))
                if metric_val is not None:
                    details.append(f"{failure}={metric_val} (umbral={thresh_val})")
                else:
                    details.append(f"{failure} (umbral={thresh_val})")
            return {
                "label": "NO_APTO_PARA_PRODUCCION",
                "status": "FAIL",
                "message": "La solución no cumple los criterios de alineación por casos.",
                "details": details,
                "recommendation": "Priorizar reducción de violaciones entre casos antes de considerar producción.",
            }

        contract_context = _summarize_contract()
        integrity_context = _summarize_integrity()
        output_contract_context = _summarize_output_contract()
        case_alignment_context = _summarize_case_alignment()
        case_alignment_business_status = _case_alignment_business_status()
        gate_context = _summarize_gate_context()
        review_feedback_context = _summarize_review_feedback()
        steward_context = _summarize_steward()
        cleaning_context = _summarize_cleaning()
        weights_context = _summarize_weights()
        case_summary_context = _summarize_case_summary()
        scored_rows_context = _summarize_scored_rows()
        run_summary_context = _summarize_run()
        data_adequacy_context = _summarize_data_adequacy()
        model_metrics_context = _summarize_model_metrics()
        facts_context = _build_fact_cards(case_summary_context, scored_rows_context, weights_context, data_adequacy_context)
        artifacts_context = artifact_index if artifact_index else []

        # Define Template
        SYSTEM_PROMPT_TEMPLATE = Template("""
        You are a Senior Executive Translator and Data Storyteller.
        Your goal is to translate technical outputs into a decision-ready business narrative.
        
        TONE: Professional, evidence-driven, decisive. Avoid unnecessary jargon.
        STYLE: Prioritize decision, evidence, risks, and next actions. No fluff.
        
        *** FORMATTING CONSTRAINTS (CRITICAL) ***
        1. **LANGUAGE:** DETECT the language of the 'Business Objective' in the state. GENERATE THE REPORT IN THAT SAME LANGUAGE. (If objective is Spanish, output Spanish).
        2. **NO MARKDOWN TABLES:** The PDF generator breaks on tables. DO NOT use table syntax. Use bulleted lists or key-value pairs instead.
           - Bad: | Metric | Value |
           - Good: 
             * Metric: Value
        
        CONTEXT:
        - Business Objective: $business_objective
        - Strategy: $strategy_title
        - Hypothesis: $hypothesis
        - Compliance Check: $compliance
        - Contract: $contract_context
        - Integrity Audit: $integrity_context
        - Output Contract: $output_contract_context
        - Case Alignment QA: $case_alignment_context
        - Business Readiness (Case Alignment): $case_alignment_business_status
        - Gate Context: $gate_context
        - Review Feedback: $review_feedback
        - Steward Summary: $steward_context
        - Cleaning Summary: $cleaning_context
        - Run Summary: $run_summary_context
        - Data Adequacy: $data_adequacy_context
        - Fact Cards (use as evidence): $facts_context
        - Model Metrics & Weights: $weights_context
        - Model Metrics (Expanded): $model_metrics_context
        - Case Summary Snapshot: $case_summary_context
        - Scored Rows Snapshot: $scored_rows_context
        - Plot Insights (data-driven): $plot_insights_json
        - Artifacts Available: $artifacts_context
        
        ERROR CONDITION:
        $error_condition
        
        VISUALS CONTEXT (JSON):
        $visuals_context_json

        IF ERROR: 
        Explain clearly what went wrong in non-technical terms and suggest next steps.
        CRITICAL: If "has_partial_visuals" is true, you MUST state: "Despite individual errors, partial visualizations were generated." and refer to the plots listed in "plots_list". Do NOT say "No visualizations created" if they exist.
        
        IF SUCCESS:
        Produce a senior-level executive report with these required sections:
        1) Executive Decision: ONE line with readiness (GO / PILOT / NO-GO) and why.
        2) Objective & Approach: What we set out to do and the approach used.
        3) Evidence & Metrics: Cite 3+ concrete numbers from Fact Cards or snapshots, with source file names.
           If a number is unavailable, write "No disponible" and state which artifact is missing.
        4) Business Impact: Translate metrics into business implications and expected value.
        5) Risks & Limitations: Call out data risks, gate failures, or misalignment with the objective.
        6) Recommended Next Actions: 2-5 specific actions (short-term + data improvements).
        7) Visual Insights: Explain what each plot shows using Plot Insights; do not describe only the chart type.

        Ensure logical consistency: do not claim elasticity, uplift, or improvements unless supported by metrics or plots.
        If quality gates are missing or misaligned, explicitly state that evaluation confidence is reduced.
        IF DATA ADEQUACY:
        If Data Adequacy indicates status "data_limited" AND threshold_reached is true,
        explicitly say the performance ceiling is likely due to data quality/coverage,
        and list 2-4 concrete data improvement steps from Data Adequacy recommendations.
        If Data Adequacy indicates "insufficient_signal", state that metrics are incomplete
        and the report should be treated as directional, not decision-grade.

        If Business Readiness indicates NO_APTO_PARA_PRODUCCION, explicitly state it and summarize
        the main reasons using gate context and review feedback in executive language.

        OUTPUT: Markdown format (NO TABLES).
        """)
        
        error_condition_str = f"CRITICAL ERROR ENCOUNTERED: {error_message}" if error_message else "No critical errors."

        system_prompt = SYSTEM_PROMPT_TEMPLATE.substitute(
            business_objective=business_objective,
            strategy_title=strategy_title,
            hypothesis=hypothesis,
            compliance=compliance,
            error_condition=error_condition_str,
            visuals_context_json=visuals_context_json,
            analysis_type=analysis_type,
            contract_context=json.dumps(contract_context, ensure_ascii=False),
            integrity_context=integrity_context,
            output_contract_context=output_contract_context,
            case_alignment_context=case_alignment_context,
            case_alignment_business_status=json.dumps(case_alignment_business_status, ensure_ascii=False),
            gate_context=json.dumps(gate_context, ensure_ascii=False),
            review_feedback=json.dumps(review_feedback_context, ensure_ascii=False),
            steward_context=json.dumps(steward_context, ensure_ascii=False),
            cleaning_context=json.dumps(cleaning_context, ensure_ascii=False),
            run_summary_context=json.dumps(run_summary_context, ensure_ascii=False),
            data_adequacy_context=json.dumps(data_adequacy_context, ensure_ascii=False),
            facts_context=json.dumps(facts_context, ensure_ascii=False),
            weights_context=json.dumps(weights_context, ensure_ascii=False),
            model_metrics_context=json.dumps(model_metrics_context, ensure_ascii=False),
            case_summary_context=json.dumps(case_summary_context, ensure_ascii=False),
            scored_rows_context=json.dumps(scored_rows_context, ensure_ascii=False),
            plot_insights_json=json.dumps(plot_insights, ensure_ascii=False),
            artifacts_context=json.dumps(artifacts_context, ensure_ascii=False)
        )
        
        # Execution Results
        execution_results = state.get('execution_output', 'No execution results available.')
        
        USER_MESSAGE_TEMPLATE = """
        Generate the Executive Report.
        
        *** EXECUTION FINDINGS (RESULTS & METRICS) ***
        $execution_results
        
        *** FULL CONTEXT (STATE) ***
        $final_state_str
        """
        
        user_message = render_prompt(
            USER_MESSAGE_TEMPLATE,
            execution_results=execution_results,
            final_state_str=str(state)
        )

        full_prompt = system_prompt + "\n\n" + user_message

        try:
            response = self.model.generate_content(full_prompt)
            return (getattr(response, "text", "") or "").strip()
        except Exception as e:
            return f"Error generating report: {e}"
