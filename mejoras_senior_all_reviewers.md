# AUDITORÍA: Agentes Reviewers de Data Engineer y ML Engineer

## Resumen Ejecutivo

El sistema de revisión está compuesto por **5 agentes** organizados en capas:

| Agente | Líneas | Revisa | Modo |
|--------|--------|--------|------|
| **CleaningReviewer** | 2651 | Salida del Data Engineer (CSV limpio + manifest) | Determinístico + LLM |
| **QAReviewer** | 1268 | Código del ML Engineer (AST estático + LLM) | Determinístico + LLM |
| **Reviewer** | 498 | Código del ML Engineer (estrategia/alineación) | Solo LLM |
| **ReviewBoard** | 140 | Veredictos consolidados de todos | LLM + fallback determinístico |
| **FailureExplainer** | 163 | Errores runtime (DE + ML) | LLM + fallback patterns |

**Veredicto global: Los reviewers son ~75% senior.** Hay una arquitectura sólida con gates contractuales + universales, merge LLM/determinístico, fail-closed, y cross-review alignment. Pero hay debilidades concretas que reducen la eficacia del feedback que reciben los ingenieros en iteraciones de repair.

---

## 1. CLEANING REVIEWER (Data Engineer)

### 1.1 Arquitectura — SÓLIDA ✅

El CleaningReviewer es el más maduro del sistema. Implementa un patrón **"deterministic-first, LLM-overlay"**:

```
Gates del contrato (cleaning_view.cleaning_gates)
        ↓
_merge_cleaning_gates() → merge con universales (_FALLBACK_CLEANING_GATES)
        ↓
_evaluate_gates_deterministic() → 11 gate types evaluados con código
        ↓
_build_llm_prompt() → LLM recibe: gates + facts + deterministic results + cleaning_code + data_profile
        ↓
_merge_llm_with_deterministic() → determinístico es canónico, LLM solo llena gaps (passed=None)
        ↓
_enforce_fail_closed_when_llm_unavailable() → si LLM falla + HARD gates sin evaluar → REJECTED
        ↓
_enforce_contract_strict_rejection() → si cleaning_gates falta del contrato → REJECTED
```

**Strengths:**
- **11 gate types determinísticos**: required_columns_present, id_integrity, no_semantic_rescale (LLM-delegated), no_synthetic_data, row_count_sanity, numeric_parsing_validation, numeric_type_casting_check, target_null_alignment_with_split, id_uniqueness_validation, null_handling_verification, outlier_policy_applied
- **Fail-closed**: Si el LLM no está disponible y hay HARD gates sin evaluar → REJECTED automáticamente
- **Contract-strict**: Si cleaning_gates falta del contrato → REJECTED con hard_failure `CONTRACT_MISSING_CLEANING_GATES`
- **Context triplet para LLM**: cleaning_code + data_profile + contract gates → razonamiento contextual, no heurísticas rígidas
- **Merge canónico**: Si determinístico dice `passed=True/False`, el LLM NO puede overridear. Solo llena gates con `passed=None`
- **Evidence requirement en prompt LLM**: "EVIDENCE: cleaning_code#line42 -> df['pixel'] = df['pixel'] / 255"
- **Dialect auto-inference**: Si el CSV cleaned tiene delimiter incorrecto, lo auto-detecta y re-lee
- **Lee 400 rows de muestra**: `sample_str` (dtype=str) + `sample_infer` (dtype=None) + `raw_sample` (200 rows)

### 1.2 Debilidades del Cleaning Reviewer

**DEBILIDAD CR-1: no_semantic_rescale 100% delegado al LLM (MEDIUM-HIGH)**

```python
elif gate_key == "no_semantic_rescale":
    issues = []
    evidence["llm_delegated"] = True
    evaluated = False  # Not deterministically evaluated
```

Este gate HARD es el único que NO tiene evaluación determinística. Si el LLM falla o no está disponible, el gate queda con `passed=None`, lo cual activa `_enforce_fail_closed` → REJECTED. Esto es correcto como safety net, pero genera falsos rechazos cuando el LLM no está disponible. Un check determinístico simple (¿hay `/255`, `MinMaxScaler`, `StandardScaler` en el código?) cubriría el 80% de los casos sin LLM.

**DEBILIDAD CR-2: _review_cleaning_failure feedback limitado (MEDIUM)**

```python
def _review_cleaning_failure(...):
    root_line = lines[-1][:300]  # Solo última línea del traceback
    
    if "numpy.ndarray" in error_details and ".str" in error_details:
        required_fixes.append("Avoid assigning np.where...")
    if "KeyError" in error_details:
        required_fixes.append("Check column name normalization...")
    if "FileNotFoundError" in error_details:
        required_fixes.append("Read the input from the provided input_path...")
    if not required_fixes:
        required_fixes.append("Fix the runtime error and rerun...")  # GENÉRICO
```

Solo 3 patrones de error conocidos + 1 fallback genérico. Cuando el DE falla por TypeError, ValueError, IndexError, o errores de pandas específicos, el fix genérico "Fix the runtime error and rerun" no da información útil al ingeniero. El FailureExplainer sí tiene más patrones pero se invoca SEPARADAMENTE en graph.py, no dentro del cleaning_reviewer.

**DEBILIDAD CR-3: Truncado de cleaning_code a 6000 chars en LLM prompt (LOW-MEDIUM)**

```python
if len(cleaning_code) > max_code_len:
    payload["cleaning_code"] = cleaning_code[:max_code_len] + "\n... [TRUNCATED]"
```

Para scripts DE complejos (>6KB), el LLM pierde la cola del código que a menudo contiene las operaciones finales de escritura y validación. Un enfoque smarter sería: head(3KB) + tail(3KB), o extraer solo funciones que tocan columnas numéricas.

**DEBILIDAD CR-4: No hay gate determinístico para tipo de encoding en output (LOW)**

Se verifica dialect (sep, decimal) con auto-inference, pero el encoding del output CSV no se valida explícitamente contra el contrato. Un CSV escrito en latin-1 cuando el contrato espera utf-8 pasará los gates actuales.

### 1.3 Feedback al Data Engineer — ROBUSTO pero con gaps

El flujo de feedback cuando CleaningReviewer rechaza es:

```python
# En graph.py línea ~14114:
if status == "REJECTED":
    if not state.get("cleaning_reviewer_retry_done"):
        fixes = review_result.get("required_fixes", [])
        fixes_text = "\nREQUIRED_FIXES:\n- " + "\n- ".join(fixes)
        payload = "CLEANING_REVIEWER_ALERT:\n" + feedback + fixes_text
        new_state["data_engineer_audit_override"] = _merge_de_audit_override(base, payload)
        return run_data_engineer(new_state)  # RETRY CON CONTEXTO
    # Si ya se reintentó → pipeline abort
```

**✅ Bueno**: El feedback incluye `required_fixes` específicas de los gates que fallaron.
**✅ Bueno**: Solo permite 1 reintento (evita loops infinitos).
**❌ Gap**: Solo 1 oportunidad. Si el primer reintento falla por otro gate, no hay segundo intento.
**❌ Gap**: El feedback NO incluye `gate_results` detallados con evidencia. Solo `feedback` (texto plano) + `required_fixes` (lista). El DE no sabe QUÉ valor tenía la columna que falló ni QUÉ gate exacto con qué parámetros.

---

## 2. QA REVIEWER (ML Engineer)

### 2.1 Arquitectura — SÓLIDA ✅

Patrón **"AST-static-first, LLM-second"**:

```
Gates del contrato (qa_view.qa_gates)
        ↓
resolve_qa_gates() → normalización + fallback a [security_sandbox, must_read_input_csv]
        ↓
run_static_qa_checks() → AST scanner + checks determinísticos
   ├── Si HARD failure → REJECTED sin llamar LLM (fast-fail)
   └── Si PASS/WARN → continúa a LLM
        ↓
LLM review con prompt + static facts + code
        ↓
Gate filter: solo falla gates que están en qa_gates
        ↓
Severity mapping: gate_lookup para HARD vs SOFT
```

**Strengths:**
- **AST Scanner exhaustivo** (`_StaticQAScanner`): Detecta security violations, forbidden imports, synthetic data, variance guards, leakage asserts, group splits, train/eval splits, read_csv calls, contract column references
- **Fast-fail**: Si static checks detectan HARD gate failure → REJECTED sin gastar LLM call
- **Gate-filtered**: LLM no puede inventar gates. Solo puede fallar gates que existen en qa_gates del contrato
- **Execution-aware**: Si existe metrics.json válido, relaja train_eval_split (downgrades a warning)
- **Plan coherence**: Valida que el código implementa lo que ml_plan.json especifica
- **Security deep check**: Detecta subprocess, os.system, shutil.rmtree, urllib patterns

**9 gate types estáticos soportados:**
1. `security_sandbox` — forbidden imports, file operations
2. `must_read_input_csv` — pd.read_csv present
3. `no_synthetic_data` — random generators, make_*, DataFrame literals
4. `must_reference_contract_columns` — canonical columns in code
5. `target_variance_guard` — nunique <= 1 check
6. `leakage_prevention` — assert_no_deterministic_target_leakage
7. `group_split_required` — GroupKFold usage when infer_group_key present
8. `train_eval_split` / `train_eval_separation` — split or CV detected
9. `plan_code_coherence` — ml_plan vs code alignment

### 2.2 Debilidades del QA Reviewer

**DEBILIDAD QA-1: No tiene gate para output artifacts (HIGH)**

No hay gate estático que verifique si el código genera los artifacts requeridos (`data/metrics.json`, `data/alignment_check.json`, `static/plots/*.png`). Esto se deja al LLM y al output_contract_report post-ejecución, pero para entonces ya se gastó un sandbox execution. Un check AST simple: ¿hay `json.dump(...)` con los paths requeridos? ¿hay `savefig(...)` si visuals están requeridos?

**DEBILIDAD QA-2: _detect_perfect_score_pattern tiene falsos positivos (MEDIUM)**

```python
if _detect_perfect_score_pattern(code) and not (scanner.has_leakage_assert or "leakage" in code.lower()):
```

Si el código legítimamente logra R2=0.99 (e.g., predicción de relaciones lineales perfectas en datos sintéticos de training), se fuerza un flag de leakage. No se cruza con los datos reales para verificar si el score perfecto es plausible.

**DEBILIDAD QA-3: AST scanner no detecta column fabrication (MEDIUM)**

El scanner detecta `pd.DataFrame(literal)` como synthetic data, pero NO detecta:
```python
df["invented_column"] = df["col_a"] + df["col_b"]  # nueva columna fuera del contrato
```
Este patrón pasa el QA y solo se detecta (quizás) por el LLM reviewer o por output schema validation.

**DEBILIDAD QA-4: Gate fallback demasiado permisivo (LOW-MEDIUM)**

```python
CONTRACT_BROKEN_FALLBACK_GATES = [
    {"name": "security_sandbox", "severity": "HARD", "params": {}},
    {"name": "must_read_input_csv", "severity": "SOFT", "params": {}},  # ← SOFT
]
```

Cuando el contrato no tiene qa_gates, el fallback solo activa 2 gates y `must_read_input_csv` es SOFT. Un ML script que no lee ningún CSV real puede pasar QA con solo warnings. Debería incluir `no_synthetic_data` como HARD en el fallback.

---

## 3. REVIEWER (ML Engineer — Estrategia/Alineación)

### 3.1 Arquitectura — FUNCIONAL pero LLM-dependiente ⚠️

```
review_code():
  LLM evalúa código contra: strategy_context + evaluation_spec + reviewer_gates + allowed_columns
  → apply_reviewer_gate_filter() → solo falla gates en reviewer_gates
  
evaluate_results():
  Phase 1: Determinístico — si "Traceback" en output → NEEDS_IMPROVEMENT sin LLM
  Phase 2: LLM evalúa resultados contra business_objective + evaluation_spec
```

**Strengths:**
- **Evidence rule**: SENIOR_EVIDENCE_RULE requiere citas concretas con artifact_path#key
- **Gate filter**: `apply_reviewer_gate_filter` impide que el LLM invente failures fuera del contrato
- **5 categorías de criterio**: Security, Methodology, Business Value, Engineering, Column Mapping
- **Execution-aware**: "If code produces valid metrics.json, methodology is likely sound"
- **Deterministic triage**: Errores runtime se detectan sin LLM call

### 3.2 Debilidades del Reviewer

**DEBILIDAD R-1: CERO checks determinísticos para review_code (HIGH)**

A diferencia de CleaningReviewer (11 gate types) y QAReviewer (9 gate types), el Reviewer no tiene NINGÚN check determinístico para `review_code()`. Todo depende del LLM. Si el LLM tiene un mal día, aprueba código con leakage o sin outputs.

Comparativa:
```
CleaningReviewer: 11 gate types determinísticos + LLM overlay
QAReviewer:        9 gate types determinísticos (AST) + LLM overlay  
Reviewer:          0 gate types determinísticos + solo LLM ← GAP
```

El `_apply_review_consistency_guard` en graph.py compensa parcialmente esto (fuerza REJECTED si hay hard_blockers de runtime/output), pero esto es post-hoc, no inline.

**DEBILIDAD R-2: evaluate_results trunca output a 4000 chars (MEDIUM)**

```python
truncated_output = execution_output[-4000:] if len(execution_output) > 4000 else execution_output
```

Solo el TAIL del output. Si el script imprime el mapping summary, métricas de baseline, y warnings al inicio, el LLM solo ve los últimos 4000 chars que típicamente son los resultados finales. Pierde contexto de warnings tempranos y baseline.

**DEBILIDAD R-3: Fail-open en error (MEDIUM)**

```python
except Exception as e:
    return {
        "status": "APPROVE_WITH_WARNINGS",  # ← APRUEBA cuando el reviewer falla
        "feedback": f"Reviewer unavailable (API error: {e})...",
    }
```

Si el LLM API falla, el Reviewer aprueba con warnings. Para un agente sin checks determinísticos, esto significa que código potencialmente malo pasa review. El CleaningReviewer hace fail-closed en esta situación (`_enforce_fail_closed_when_llm_unavailable`). El Reviewer debería hacer lo mismo.

**DEBILIDAD R-4: No recibe el código anterior en patch mode (LOW-MEDIUM)**

En `review_code()`, solo recibe el código actual. No recibe el diff vs. iteración anterior. El LLM no puede evaluar si el patch preservó funcionalidad existente o si rompió algo que antes funcionaba.

---

## 4. REVIEW BOARD (Adjudicador Final)

### 4.1 Arquitectura — FUNCIONAL ✅

```
board_context = {reviewer, qa_reviewer, result_evaluator, deterministic_facts, runtime}
        ↓
LLM adjudica → verdict
        ↓
Post-processing: deterministic_blockers override → si hay blockers + LLM dijo APPROVED → NEEDS_IMPROVEMENT
```

**Strengths:**
- **deterministic_facts son canónicos**: Si LLM contradice facts, se prioriza facts
- **fallback determinístico**: Si LLM falla, usa logic basada en statuses de los 3 reviewers
- **Blocker escalation**: Deterministic blockers SIEMPRE prevalecen sobre LLM APPROVED
- **Runtime terminal awareness**: Si runtime fix es terminal (max retries), ajusta areas críticas

### 4.2 Debilidades del Review Board

**DEBILIDAD RB-1: 140 líneas = oversimplified (MEDIUM)**

Con solo 140 líneas, el Review Board no tiene:
- Reconciliación de conflictos entre Reviewer y QA (e.g., Reviewer dice APPROVED, QA dice REJECTED)
- Peso diferencial para gates HARD vs SOFT en decisión final
- Tracking de progreso entre iteraciones (¿mejoró la métrica?)

La `_harmonize_review_packets_with_final_eval` en graph.py compensa esto parcialmente.

**DEBILIDAD RB-2: Fallback pierde required_fixes (LOW-MEDIUM)**

```python
def _fallback(self, context):
    return {
        "required_actions": ["Apply reviewer-required fixes and rerun."] if status in {...} else [],
        # ← NO propaga los required_fixes específicos de reviewer/QA
    }
```

El fallback genera `required_actions` genéricas. No propaga las `required_fixes` concretas de los packets de reviewer y QA.

---

## 5. FAILURE EXPLAINER

### 5.1 Arquitectura — FUNCIONAL pero limitada ⚠️

**Strengths:**
- Formato estructurado: WHERE/WHY/FIX
- Contexto incluye: code (6KB) + error (4KB) + context (2KB)
- Retries: `call_with_retries(max_retries=2)`
- Fallback patterns: 4 patrones determinísticos

**DEBILIDAD FE-1: Solo 4 fallback patterns (HIGH)**

```python
def _fallback(self, error_details):
    if "list of cases must be same length..." → np.select mismatch
    if "numpy.bool_" and "not serializable" → json.dumps numpy
    if "keyerror" and "not in index" → column missing after rename
    if "missing required columns" → columns missing
    return ""  # ← VACÍO para todo lo demás
```

Si el LLM no está disponible, el FailureExplainer devuelve string vacío para la mayoría de errores. Esto significa que en DE retry y ML retry, el campo `LLM_FAILURE_EXPLANATION` del feedback queda vacío, y el ingeniero solo tiene el traceback raw.

---

## 6. SISTEMA DE FEEDBACK — Análisis de Robustez

### 6.1 Flujo de Feedback para Data Engineer

```
Runtime Error → _infer_de_failure_cause() + _build_de_runtime_diagnosis() 
             + _build_de_gate_implementation_hints() + FailureExplainer.explain_de_failure()
             → Todo concatenado en data_engineer_audit_override
             → DE recibe como contexto en siguiente iteración

Cleaning Review REJECTED → feedback + required_fixes 
             → CLEANING_REVIEWER_ALERT en audit_override
             → 1 reintento permitido
```

**FORTALEZAS del feedback DE:**
- ✅ `_infer_de_failure_cause`: 12 patrones de error reconocidos con causas raíz
- ✅ `_build_de_runtime_diagnosis`: 7 diagnósticos expandidos con instrucciones específicas
- ✅ `_build_de_gate_implementation_hints`: Genera hints por gate type-cast con scope y dtype
- ✅ FailureExplainer: LLM analiza code+traceback+context → WHERE/WHY/FIX
- ✅ Error truncado: `error_details[-2000:]` preserva la cola (donde está el error real)

**DEBILIDADES del feedback DE:**

**DEBILIDAD FB-DE-1: audit_override es texto plano acumulativo (HIGH)**

```python
new_state["data_engineer_audit_override"] = override  
# override = base + "\n\nRUNTIME_ERROR_CONTEXT:\n" + error[-2000:]
#          + "\n\n" + reviewer_payload 
#          + "\nWHY_IT_HAPPENED: " + cause
#          + "\nERROR_DIAGNOSIS:\n- " + diagnosis
#          + "\nGATE_IMPLEMENTATION_HINTS:\n- " + hints
#          + "\nLLM_FAILURE_EXPLANATION:\n" + explainer
```

Todo se concatena como texto plano en `data_engineer_audit_override`. Después de 2-3 iteraciones, este campo acumula múltiples bloques de error de iteraciones anteriores, mezclados, sin separación clara de qué es de cuál iteración. El LLM del DE puede confundirse con errores obsoletos de iteraciones previas.

**DEBILIDAD FB-DE-2: No hay evidencia de sample de datos en feedback (MEDIUM)**

Cuando el CleaningReviewer detecta que `id_integrity` falla porque una columna ID tiene notación científica, el `evidence` dict contiene la información pero NO se incluye en el `required_fixes` que llega al DE. El DE recibe "Check column name normalization" pero no sabe qué valor problemático causó el fallo.

### 6.2 Flujo de Feedback para ML Engineer

```
Review REJECTED → gate_context = {source, status, feedback, failed_gates, required_fixes, hard_failures}
              + _expand_required_fixes() → enrichment con guidance_map (40+ patterns)
              + _build_fix_instructions() → "PATCH TARGETS: - fix1 - fix2"
              → last_gate_context en state
              
QA REJECTED → same structure pero desde static checks + LLM
              
Runtime Error → evaluate_results() Phase 1 → deterministic triage
              → required_fixes para errores conocidos

Iteration Handoff → _build_iteration_handoff() → structured JSON:
    {contract_focus, quality_focus, metric_focus, feedback, must_preserve, patch_objectives}
```

**FORTALEZAS del feedback ML:**
- ✅ `_expand_required_fixes`: **40+ patterns** en guidance_map con instrucciones muy específicas (e.g., DIALECT_LOADING_MISSING tiene 11 líneas de instrucción con código de ejemplo)
- ✅ `_build_iteration_handoff`: Handoff estructurado con 7 secciones claras (contract/quality/metric/feedback/must_preserve/patch_objectives)
- ✅ `must_preserve`: Lista de artifacts que el patch debe mantener intactos
- ✅ `metric_focus`: Primary metric + baseline + target + gap → el ML sabe exactamente cuánto mejorar
- ✅ Cross-review alignment: `_harmonize_review_packets_with_final_eval` sincroniza reviewer/QA con evaluator
- ✅ `_apply_review_consistency_guard`: Si hay deterministic blockers + reviewer aprobó → fuerza REJECTED

**DEBILIDADES del feedback ML:**

**DEBILIDAD FB-ML-1: iteration_handoff trunca feedback (HIGH)**

```python
reviewer_feedback = _truncate_handoff_text(review_result.get("feedback") or ..., max_len=520)
qa_feedback = _truncate_handoff_text(qa_result.get("feedback"), max_len=360)
runtime_tail = _truncate_handoff_text(state.get("last_runtime_error_tail") or ..., max_len=360)
```

El feedback del reviewer se trunca a **520 chars**, QA a **360 chars**, runtime a **360 chars**. Cuando un reviewer da feedback detallado ("You need to fix X because Y, evidence: Z, and also check W because..."), se corta. El ML engineer recibe feedback incompleto. Para un reviewer "senior", el output debería preservar el feedback completo o al menos las `required_fixes` sin truncar (que sí se preservan por separado con max_len=240 per fix, max_items=15).

**DEBILIDAD FB-ML-2: feedback_history acumula texto bruto (MEDIUM)**

```python
new_history.append(f"REVIEWER FEEDBACK (Attempt {current_iter+1}): {review['feedback']}")
```

Similar al problema del DE: feedback_history es una lista de strings que crece con cada iteración. El ML engineer recibe `_select_feedback_blocks(max_blocks=2)` que toma los últimos 2 bloques, pero estos bloques no tienen estructura — son texto plano del LLM reviewer.

**DEBILIDAD FB-ML-3: Reviewer y QA no comparten findings (MEDIUM)**

El Reviewer y el QA Reviewer evalúan el MISMO código pero independientemente. Sus findings no se reconcilian antes de llegar al ML engineer:
- Reviewer dice "fix leakage" → gate_context con failed_gates=["leakage"]
- QA dice "fix synthetic_data" → gate_context se SOBREESCRIBE o no se merge
- ML engineer puede recibir solo uno de los dos

El `_build_iteration_handoff` intenta merge, pero toma `gate_context.get("failed_gates")` del último reviewer que escribió en state. Si QA escribió después del Reviewer, se pierden los failed_gates del Reviewer.

---

## 7. TABLA DE EVALUACIÓN: ¿Son Senior?

| Criterio Senior | CleaningReviewer | QAReviewer | Reviewer | ReviewBoard |
|----------------|:---:|:---:|:---:|:---:|
| Gates universales (fallback) | ✅ 5 gates | ✅ 2 gates* | ❌ | N/A |
| Gates contractuales | ✅ 11 types | ✅ 9 types | ✅ filtered | ✅ |
| Checks determinísticos | ✅ 11 | ✅ 9 (AST) | ❌ 0 | ✅ fallback |
| LLM contextual | ✅ triplet | ✅ code audit | ✅ strategy | ✅ adjudication |
| Merge det+LLM | ✅ canónico | ✅ fast-fail | ❌ N/A | ✅ blockers override |
| Fail-closed | ✅ | ⚠️ partial | ❌ fail-open | ✅ fallback |
| Evidence requirement | ✅ explicit | ✅ explicit | ✅ explicit | ✅ explicit |
| Feedback específico | ⚠️ limitado | ✅ guidance_map | ⚠️ truncado | ⚠️ genérico |
| Awareness iteración previa | ❌ | ⚠️ via handoff | ⚠️ via handoff | ❌ |

\* Fallback QA solo tiene 2 gates (security_sandbox HARD + must_read_input_csv SOFT)

---

## 8. PROPUESTAS DE MEJORA — Priorizadas

### PRIORIDAD 1: Quick Wins (1-2 horas cada uno)

**P1.1: Reviewer fail-closed cuando LLM no disponible**
```python
# reviewer.py review_code() - cambiar de fail-open a fail-closed
except Exception as e:
    return {
        "status": "REJECTED",  # ERA: "APPROVE_WITH_WARNINGS"
        "feedback": f"Reviewer LLM unavailable: {e}. Cannot approve without review.",
        "failed_gates": ["LLM_REVIEW_UNAVAILABLE"],
        "required_fixes": ["Retry when reviewer LLM is available."],
    }
```
**Impacto**: Elimina riesgo de código malo pasando review por falla de API.

**P1.2: Feedback no-truncado para required_fixes en handoff**
```python
# graph.py _build_iteration_handoff() - preservar fixes completas
required_fixes = _normalize_handoff_items(
    gate_context.get("required_fixes"), 
    max_items=15, 
    max_len=500  # ERA: 240
)
reviewer_feedback = _truncate_handoff_text(
    review_result.get("feedback") or ..., 
    max_len=1200  # ERA: 520
)
```
**Impacto**: ML engineer recibe instrucciones completas de fix.

**P1.3: Agregar no_synthetic_data HARD al QA fallback**
```python
CONTRACT_BROKEN_FALLBACK_GATES = [
    {"name": "security_sandbox", "severity": "HARD", "params": {}},
    {"name": "must_read_input_csv", "severity": "HARD", "params": {}},  # ERA: SOFT
    {"name": "no_synthetic_data", "severity": "HARD", "params": {}},  # NUEVO
]
```
**Impacto**: Cuando el contrato no tiene qa_gates, synthetic data se rechaza automáticamente.

**P1.4: no_semantic_rescale con check determinístico parcial**
```python
# cleaning_reviewer.py - agregar check determinístico básico
elif gate_key == "no_semantic_rescale":
    code_lower = (cleaning_code or "").lower()
    rescale_patterns = ["/255", "/ 255", "minmaxscaler", "standardscaler", 
                        "* 0.00392", "normalize("]
    found = [p for p in rescale_patterns if p in code_lower]
    if found:
        issues = [f"Rescaling pattern detected in code: {', '.join(found)}"]
        evidence["patterns_found"] = found
        evaluated = True
    else:
        evidence["llm_delegated"] = True
        evaluated = False  # Delegate to LLM
```
**Impacto**: Catch los rescalings más obvios sin LLM. LLM sigue evaluando casos ambiguos.

**P1.5: FailureExplainer con más fallback patterns**
```python
def _fallback(self, error_details):
    # ... patterns existentes ...
    if "typeerror" in lower and "not supported between" in lower:
        return "Type comparison on mixed types; convert to numeric before comparison."
    if "valueerror" in lower and "could not convert" in lower:
        return "String-to-numeric conversion failed; use pd.to_numeric(errors='coerce')."
    if "indexerror" in lower:
        return "Array/list index out of range; check lengths before indexing."
    if "attributeerror" in lower and "has no attribute" in lower:
        return "Method called on wrong type; verify object type before method call."
    if "filenotfounderror" in lower:
        return "File path incorrect; use contract-provided paths."
    if "memoryerror" in lower or "killed" in lower:
        return "Out of memory; reduce dataset size or use chunked processing."
    if "importerror" in lower or "modulenotfounderror" in lower:
        return "Missing dependency; use only allowed packages from runtime allowlist."
    return ""
```
**Impacto**: Cuando LLM no disponible, 11 patterns → ~60% de errores cubiertos vs 4 patterns actuales.

### PRIORIDAD 2: Esfuerzo Medio (medio día cada uno)

**P2.1: Feedback estructurado por iteración (no acumulativo)**

Reemplazar text plano acumulativo por JSON estructurado:
```python
# En graph.py, crear iteration_journal por agent:
iteration_record = {
    "iteration": attempt_id,
    "status": "REJECTED",
    "gate_results": [{name, severity, passed, issues, evidence}],
    "required_fixes": [...],
    "runtime_error": error_details[-2000:],
    "failure_diagnosis": diagnosis_lines,
    "explainer": explainer_text,
}
# Solo pasar la ÚLTIMA iteration_record al engineer, no todo el historial
```
**Impacto**: Engineer recibe feedback limpio de una sola iteración, sin ruido de iteraciones previas.

**P2.2: Merge de findings Reviewer + QA antes del handoff**

```python
def _merge_reviewer_qa_findings(reviewer_result, qa_result):
    merged_gates = list(set(
        (reviewer_result.get("failed_gates") or []) + 
        (qa_result.get("failed_gates") or [])
    ))
    merged_fixes = _dedupe_list(
        (reviewer_result.get("required_fixes") or []) + 
        (qa_result.get("required_fixes") or [])
    )
    merged_hard = list(set(
        (reviewer_result.get("hard_failures") or []) +
        (qa_result.get("hard_failures") or [])
    ))
    return {"failed_gates": merged_gates, "required_fixes": merged_fixes, "hard_failures": merged_hard}
```
**Impacto**: ML engineer recibe findings consolidados, no solo los del último reviewer que escribió en state.

**P2.3: Reviewer con checks determinísticos mínimos**

Añadir al Reviewer al menos 3-4 checks determinísticos pre-LLM:
1. **Syntax check**: `ast.parse(code)` — si falla, REJECTED sin LLM
2. **Output path check**: ¿el código genera los paths requeridos por evaluation_spec?
3. **Data load check**: ¿hay `pd.read_csv` con el `data_path` correcto?
4. **Target variable check**: ¿el target del contrato aparece en el código?

**Impacto**: Reviewer ya no es 100% LLM-dependiente.

### PRIORIDAD 3: Esfuerzo Mayor (1+ día)

**P3.1: Gate evidence propagation al engineer**

Cuando un gate falla con evidencia específica (e.g., `id_integrity` con `evidence.scientific_notation_columns: ["col_a"]`), propagar esa evidencia como contexto al engineer:
```
GATE_FAILURE_EVIDENCE:
- Gate: id_integrity (HARD)
  Issue: Scientific notation detected in ID column
  Evidence: column="customer_id", sample_values=["1.23e+05", "4.56e+07"]
  Fix: Use .astype(str) to prevent scientific notation in ID columns
```

**P3.2: Review Board con reconciliación explícita**

Expandir ReviewBoard para que compare explícitamente reviewer vs QA findings y resuelva conflictos:
- Si Reviewer APPROVED pero QA REJECTED → investigar si el QA gate es válido
- Si ambos REJECTED pero por gates diferentes → merge todos los gates
- Tracking de progreso: ¿cuántos gates pasaron vs iteración anterior?

---

## 9. RESUMEN DE ASSESSMENT

| Componente | Nivel Actual | Con P1 | Con P1+P2 |
|-----------|:---:|:---:|:---:|
| CleaningReviewer | **85% Senior** | 90% | 95% |
| QAReviewer | **80% Senior** | 88% | 93% |
| Reviewer | **55% Senior** | 65% | 80% |
| ReviewBoard | **70% Senior** | 70% | 82% |
| FailureExplainer | **50% Senior** | 70% | 75% |
| Feedback DE | **72% Robusto** | 80% | 90% |
| Feedback ML | **68% Robusto** | 78% | 88% |
| **Sistema Global** | **~70% Senior** | **~78%** | **~87%** |

**Bottom line**: El CleaningReviewer y QAReviewer son los más maduros gracias a su dual-mode (determinístico + LLM). El Reviewer (ML strategy) es el eslabón más débil por su dependencia total del LLM y su fail-open. El sistema de feedback es arquitecturalmente correcto pero pierde fidelidad por truncado, acumulación de texto plano, y falta de merge entre reviewers. Las mejoras P1 (quick wins) elevarían el sistema de ~70% a ~78% senior; P1+P2 lo llevarían a ~87%.