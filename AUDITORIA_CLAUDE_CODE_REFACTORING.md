# 🔎 AUDITORÍA DEL REFACTORING DE CLAUDE CODE
## Basado en la Auditoría de Seniority de Agentes
**Fecha:** 2026-02-15  
**Auditor:** Antigravity (Gemini)  
**Trabajo auditado:** Refactoring de Claude Code en 12 fases

---

## 📊 RESUMEN EJECUTIVO

| Fase | Recomendación Original | ¿Implementado? | ¿Correcto? | Riesgo |
|---|---|---|---|---|
| **F1: Execution Planner - Token Sets** | Eliminar 7 token sets hardcodeados | ✅ Sí | ⚠️ Parcial | 🟡 Medio |
| **F2: Failure Explainer - Fallback** | Eliminar 12 patrones hardcodeados | ✅ Sí | ✅ Correcto | 🟢 Bajo |
| **F3: Cleaning Reviewer - Gates/regex** | Eliminar fallback gates y regex | ✅ Sí | ⚠️ Parcial | 🟡 Medio |
| **F4: QA Reviewer - Keywords** | Eliminar _REGRESSOR_KEYWORDS | ✅ Sí | ⚠️ Parcial | 🟡 Medio |
| **F6: Steward - Token sets** | Eliminar SPLIT/TEMPORAL tokens | ✅ Sí | ✅ Correcto | 🟢 Bajo |
| **F7: domain_knowledge.py** | Eliminar archivo | ✅ Sí | ✅ Correcto | 🟢 Bajo |
| **F8: Domain Expert - LLM-primary** | LLM reviews con prioridad | ✅ Sí | ✅ Correcto | 🟢 Bajo |
| **F9: Import blocklist DE/ML** | Agregar SANDBOX SECURITY sección | ✅ Sí | ✅ Correcto | 🟢 Bajo |
| **F10: Review Board - Contexto hist.** | Iteration history + plateau | ✅ Sí | ✅ Correcto | 🟢 Bajo |
| **F11: graph.py - Módulos extraídos** | Extraer funciones a steps/ | ⚠️ Parcial | 🔴 Incompleto | 🔴 Alto |
| **F12: Retry loop mejorado** | _build_retry_context() | ✅ Sí | ✅ Correcto | 🟢 Bajo |
| **Tests** | No romper tests existentes | ✅ Sí | ✅ Correcto | 🟢 Bajo |

**Veredicto global: 8 de 12 fases correctamente implementadas, 3 parciales, 1 incompleta con riesgo alto.**

---

## 📋 ANÁLISIS DETALLADO POR FASE

---

### ✅ Fase 2: Failure Explainer — CORRECTO

**Lo que hizo Claude Code:**
- Eliminó los 12 patrones if/elif hardcodeados del `_fallback()`.
- Reemplazó con: `f"Automated diagnosis unavailable. Raw error summary: {error_details[:500]}"`

**Veredicto:** ✅ **Bien hecho.** Exactamente lo que recomendé. El fallback ahora pasa el error raw al downstream en vez de intentar diagnosticarlo con pattern matching. Los prompts LLM de `explain_data_engineer_failure` y `explain_ml_failure` se mantienen intactos y son la fuente primaria de diagnóstico.

**Riesgo:** 🟢 Bajo.

---

### ✅ Fase 6: Steward — CORRECTO

**Lo que hizo Claude Code:**
- Eliminó `SPLIT_CANDIDATE_TOKENS` y `_TEMPORAL_HINT_TOKENS` como constantes globales.
- Las reemplazó con `_split_hints` y `_temporal_hints` como variables **locales** dentro de las funciones que las usan.
- Añadió detección dtype-based para temporales: `pd.api.types.is_datetime64_any_dtype(df[col])`.

**Veredicto:** ✅ **Correcto y bien razonado.** Las constantes globales se convirtieron en hints locales, y se añadió detección por dtype como complemento. El comentario es preciso: "kept as minimal structural hints only for the evidence-layer profiling, not for classification decisions."

**Riesgo:** 🟢 Bajo. Los tokens son idénticos pero ahora son locales, lo cual es mejor encapsulación. La detección por dtype es una mejora real.

---

### ✅ Fase 7: domain_knowledge.py eliminado — CORRECTO

**Lo que hizo Claude Code:**
- Eliminó `src/utils/domain_knowledge.py`.
- Removió `from src.utils.domain_knowledge import infer_domain_guidance` del `domain_expert.py`.
- Eliminó la llamada a `infer_domain_guidance()` y el parámetro `domain_guidance` del `_build_prompt()`.
- Actualizó el prompt section de `*** DOMAIN GUIDANCE ***` con instrucciones para que el LLM infiera domain knowledge:
  ```
  Infer domain-specific best practices and risks from the data context and business objective.
  Do not rely on pre-defined domain templates.
  ```

**Veredicto:** ✅ **Correcto.** La eliminación del archivo de conocimiento de dominio pre-definido es exactamente lo que recomendé. El prompt ahora instruye al LLM a razonar sobre el dominio en vez de usar templates fijos.

**Riesgo:** 🟢 Bajo.

---

### ✅ Fase 8: Domain Expert — LLM-primary scoring — CORRECTO

**Lo que hizo Claude Code:**
```python
# Antes: deterministic reviews tenían prioridad por defecto
merged: Dict[int, Dict] = {int(r["strategy_index"]): dict(r) for r in deterministic_reviews}
for item in normalized_llm:
    merged[idx] = item  # LLM sobreescribe

# Ahora: LLM tiene prioridad explícita
llm_by_idx = {idx: item for item in normalized_llm}  # LLM primero
det_by_idx = {int(r["strategy_index"]): dict(r) for r in deterministic_reviews}
# Usa LLM si existe, deterministic solo como safety net
if idx in llm_by_idx:
    base = llm_by_idx[idx]
elif idx in det_by_idx:
    base = det_by_idx[idx]
```

**Veredicto:** ✅ **Correcto.** El cambio de prioridad es sutil pero importante. Antes, el dict `merged` empezaba con deterministic y luego sobreescribía con LLM (lo cual técnicamente también daba prioridad al LLM, pero la intención no era clara). Ahora la prioridad es explícita y legible: LLM primero, deterministic solo como fallback.

**Riesgo:** 🟢 Bajo.

---

### ✅ Fase 9: Import blocklist en DE/ML — CORRECTO

**Lo que hizo Claude Code:**
Agregó esta sección idéntica en ambos prompts (DE y ML):
```
SANDBOX SECURITY - BLOCKED IMPORTS (HARD CONSTRAINT):
These imports are FORBIDDEN and will cause immediate script rejection:
- sys, subprocess, socket, requests, httpx, urllib, ftplib
- paramiko, selenium, playwright, openai, google.generativeai, builtins
- eval(), exec(), compile(), __import__()
ALLOWED imports: pandas, numpy, sklearn, scipy, xgboost, catboost, lightgbm,
matplotlib, seaborn, json, os.path, os.makedirs, csv, math, statistics,
collections, itertools, functools, typing, warnings, re, datetime, pathlib.Path
If you need sys.stdout or sys.exit, use print() and raise SystemExit instead.
```

**Veredicto:** ✅ **Excelente.** Esta es probablemente la mejora de mayor impacto inmediato — directamente aborda el root cause de la run fallida `ad3ee87d` donde tanto el DE como el ML generaron `import sys`. La lista de imports prohibidos es explícita, la allowlist es completa, y la alternativa para `sys` está documentada.

**Riesgo:** 🟢 Bajo.

---

### ✅ Fase 10: Review Board — Contexto histórico — CORRECTO

**Lo que hizo Claude Code:**
1. Añadió instrucciones al prompt del Board:
   ```
   6) Use iteration_history when available to detect progress trends, plateaus, or regressions across iterations.
      If metrics have plateaued for 2+ iterations with no improvement, flag it in required_actions.
   ```
2. Añadió plateau detection en `_fallback()` (líneas 172-186): compara métricas de las últimas 2 iteraciones.
3. En `graph.py`, `run_review_board` ahora construye `iteration_history` desde `metric_history` y lo pasa al `board_context`.

**Veredicto:** ✅ **Bien implementado.** La detección de plateau es simple pero efectiva (abs diff < 1e-6). El contexto histórico se limita a las últimas 10 iteraciones para no saturar el prompt.

**Riesgo:** 🟢 Bajo.

---

### ✅ Fase 12: Retry loop mejorado — CORRECTO

**Lo que hizo Claude Code:**
Creó `_build_retry_context()` con clasificación estructurada de errores:
```python
{
    "error_type": "security_violation" | "runtime_error" | "output_missing" | "gate_failure" | "unknown",
    "specific_error": str,            # Error truncado a 500 chars
    "blocked_imports": List[str],     # Imports que causaron el rechazo
    "missing_outputs": List[str],     # Outputs que faltan
    "working_components": List[str],  # Lo que SÍ funcionó
    "failed_gates": List[Dict],       # Gates que fallaron con evidencia
}
```

El `retry_context` se integra en `_build_iteration_handoff()` como campo adicional.

**Veredicto:** ✅ **Bien diseñado.** La clasificación de error types es crucial para que el ML/DE sepan exactamente qué arreglar. El campo `working_components` previene el "Retry Amnesia" que identifiqué. La extracción de `blocked_imports` desde `last_safety_scan` es un buen detalle.

**Riesgo:** 🟢 Bajo.

---

### ⚠️ Fase 1: Execution Planner — Token Sets — PARCIAL

**Lo que hizo Claude Code:**
- Eliminó 7 token sets globales (~120 tokens total).
- Eliminó funciones helper: `_matches_any_phrase`, `_contains_decisioning_token`, `_contains_visual_token`.
- Creó `CAPABILITY_DETECTION_PROMPT` con instrucciones para detección semántica por LLM.
- Inyectó el prompt en `CONTRACT_HARD_RULES`.

**⚠️ Problemas detectados:**

1. **`_strategy_mentions_resampling` reintroduce tokens inline:**
   ```python
   # Antes: return any(token in haystack for token in _RESAMPLING_TOKENS)
   # Ahora:
   return any(tok in haystack for tok in ("resamp", "cross valid", "cross-valid", "kfold", "k-fold", "bootstrap", "stratified"))
   ```
   Se eliminaron como constantes globales pero se reintrodujeron como tupla literal inline. Los tokens cambiaron ligeramente (usando prefijos como "resamp" en vez de "resampling" y "resample"), lo cual es más genérico, pero sigue siendo una lista hardcodeada.

2. **`_build_visual_requirements` reintroduce tokens inline:**
   ```python
   enabled = bool(vision_text and any(tok in vision_text.split() for tok in ("visual", "plot", "chart", "graph", "diagram", "figure")))
   ```
   Se eliminó `_VISUAL_ENABLED_TOKENS` (21 tokens) pero se reemplazó con una lista más corta de 6 tokens. Mejora parcial — la lista es más pequeña y genérica, pero sigue siendo keyword matching.

3. **`_build_decisioning_requirements` mantiene keyword matching:**
   ```python
   enabled = bool(existing_dec.get("enabled")) or any(
       kw in objective_type for kw in ["rank", "priority", "decision", "segment", "triage", "outlier", "action"]
   )
   ```
   Se consultó el contrato LLM-generado como fuente primaria (`existing_dec.get("enabled")`), lo cual es correcto, pero mantiene un fallback con keywords. Esto es razonable — el CAPABILITY_DETECTION_PROMPT le dice al LLM que setee los flags, y las keywords son fallback para cuando el LLM no los setea.

4. **CAPABILITY_DETECTION_PROMPT aparece duplicado:** Una vez como constante `CAPABILITY_DETECTION_PROMPT` (línea ~140) y otra vez inline dentro de `CONTRACT_HARD_RULES` (línea ~167). Es redundante.

**Veredicto:** ⚠️ **Parcialmente correcto.** La dirección es buena (LLM como fuente primaria, keywords como fallback), pero los tokens no se eliminaron realmente — se reorganizaron. La mejora real es que ahora se consulta el contrato LLM-generado primero, y las keywords son fallback más cortas y genéricas.

**Riesgo:** 🟡 Medio. Las funciones siguen funcionando correctamente, pero el "cero hardcoding" no se logró del todo.

---

### ⚠️ Fase 3: Cleaning Reviewer — Gates/regex — PARCIAL

**Lo que hizo Claude Code:**
- Eliminó `_FALLBACK_CLEANING_GATES` (6 gates, ~35 líneas).
- Eliminó `_DEFAULT_ID_REGEX` y `_DEFAULT_PERCENT_REGEX`.
- Reemplazó con `_GENERIC_ID_REGEX` más corto (sin "partida", "invoice", "account", "plazo").
- `_merge_cleaning_gates()` ahora retorna lista vacía si no hay gates en el contrato.
- Eliminó `alias_map` de `_normalize_gate_name()`.

**⚠️ Problemas detectados:**

1. **`_GENERIC_ID_REGEX` sigue siendo una regex hardcodeada:**
   ```python
   _GENERIC_ID_REGEX = r"(?i)(^id$|(?:_id$)|(?:^id_)|(?:^|[_\W])(?:id|entity|code|key)(?:[_\W]|$))"
   ```
   Es más genérica que antes (se eliminaron los tokens en español), pero sigue siendo pattern matching por nombre. La recomendación era usar `column_roles` del contrato como fuente primaria, y solo caer a regex si no hay roles. La implementación actual ya consulta `column_roles` primero (via `_columns_with_role_tokens`), lo cual es correcto, pero la regex genérica como fallback es aceptable.

2. **`_DEFAULT_PERCENT_REGEX` NO fue reemplazada:** Se eliminó la constante pero no veo un reemplazo genérico en el diff. Si alguna función la usaba como regex de fallback para detectar columnas porcentuales, podría faltar. Sin embargo, el gate `no_semantic_rescale` que la usaba fue eliminado con los fallback gates, así que esto probablemente no es un problema.

3. **La eliminación del `alias_map` puede causar regresión funcional:** Antes, variantes como "numeric_parsing_verification" se mapeaban a "numeric_parsing_validation". El test fue actualizado para reflejar esto (`assert _normalize_gate_name("Numeric Parsing Verification") == "numeric_parsing_verification"` en vez de `"numeric_parsing_validation"`). Esto significa que un contrato que genere "numeric_parsing_verification" ya no se reconciliará con gates que esperan "numeric_parsing_validation". **Esto puede causar gate mismatches silenciosos.** Sin embargo, si el contrato V4.1 es la fuente de verdad para los nombres de gates, y tanto el productor como el consumidor usan el mismo nombre, esto debería ser consistente.

**Veredicto:** ⚠️ **Parcialmente correcto.** Las eliminaciones principales son correctas. La regex genérica es aceptable como fallback después de `column_roles`. La eliminación del alias map es riesgosa pero coherente con la filosofía de "el contrato es la fuente de verdad".

**Riesgo:** 🟡 Medio. Posibles gate mismatches si el LLM genera nombres de gates con variantes que antes se normalizaban.

---

### ⚠️ Fase 4: QA Reviewer — Keywords — PARCIAL

**Lo que hizo Claude Code:**
- Eliminó `_REGRESSOR_KEYWORDS` (16 nombres hardcodeados).
- Eliminó `CONTRACT_BROKEN_FALLBACK_GATES`.
- `_looks_like_regressor()` ahora usa convención de nombres + 4 excepciones hardcodeadas.
- `resolve_qa_gates()` retorna lista vacía si faltan gates.

**⚠️ Problemas detectados:**

1. **`_looks_like_regressor` NO es realmente "genérico por convención":**
   ```python
   if simple.endswith("Regressor"): return True     # ✅ Genérico
   if simple.lower() in {"svr", "linearsvr"}: ...   # ❌ Hardcodeado
   if simple in {"ElasticNet", "Lasso", "Ridge", "LinearRegression"}: ... # ❌ Hardcodeado
   ```
   Se eliminó el set de 16 nombres y se reemplazaron con: 1 convención genérica (`endswith("Regressor")`) + 6 nombres aún hardcodeados. Es una mejora parcial — pasó de 16 a 6 nombres fijos. Sin embargo, Claude Code documentó por qué: "common regression models inherit RegressorMixin", que es una justificación válida para mantenerlos como excepciones a la convención.

2. **`resolve_qa_gates()` retorna lista vacía es potencialmente peligroso:** Si no hay gates y retorna `[]`, el QA reviewer no evalúa ningún gate. Esto podría dejar pasar código sin revisión de calidad. La implementación anterior usaba `CONTRACT_BROKEN_FALLBACK_GATES` con `security_sandbox`, `must_read_input_csv`, y `no_synthetic_data` como safety net. Sin estos gates de seguridad, el QA reviewer pierde su última línea de defensa.

**Veredicto:** ⚠️ **Parcialmente correcto.** La reducción de 16 a 6 nombres en `_looks_like_regressor` es buena dirección pero no es "cero hardcoding". La eliminación de `CONTRACT_BROKEN_FALLBACK_GATES` es la más arriesgada — debería al menos mantener `security_sandbox` como gate incondicional.

**Riesgo:** 🟡 Medio. La ausencia de gates de seguridad cuando el contrato falla es un riesgo real.

---

### 🔴 Fase 11: graph.py — Módulos extraídos — INCOMPLETO / CON PROBLEMAS

**Lo que hizo Claude Code:**
Creó `src/graph/steps/` con 3 módulos:
- `contract_resolution.py` (322 líneas, 7 funciones)
- `context_builders.py` (684 líneas, 5 funciones públicas + helpers)
- `result_evaluator.py` (158 líneas, 3 funciones)

E importó con aliases en graph.py:
```python
from src.graph.steps.contract_resolution import (
    _resolve_contract_columns as _steps_resolve_contract_columns,
    ...
)
```

**🔴 PROBLEMAS CRÍTICOS DETECTADOS:**

1. **Las funciones están DUPLICADAS, no EXTRAÍDAS:**
   Las funciones originales **siguen existiendo en `graph.py`** (confirmado: `_resolve_contract_columns` en línea 2804, `_build_cleaned_data_summary_min` en línea 2455, `_apply_review_consistency_guard` en línea 1048, `_harmonize_review_packets_with_final_eval` en línea 1098). Claude Code creó copias en los módulos `steps/` pero NO eliminó los originales de `graph.py`.

2. **Los imports con alias `_steps_*` NUNCA SE USAN:**
   Busqué `_steps_resolve_contract_columns`, `_steps_build_cleaned_data_summary_min`, `_steps_apply_review_consistency_guard`, `_steps_harmonize_review_packets`, `_steps_looks_blocking_retry_signal`, `_steps_build_required_sample_context`, `_steps_build_signal_summary_context` en graph.py. **Ninguno tiene resultados** — son imports muertos.

3. **`graph.py` sigue teniendo +21K líneas:**
   El diff muestra **114 líneas añadidas y 0 eliminadas** en graph.py. Las funciones extraídas suman ~1,150 líneas que deberían haberse eliminado de graph.py. El archivo NO se redujo.

4. **`_norm_name` existe en 3 copias:**
   - `graph.py` línea 265
   - `context_builders.py` línea 40
   - `contract_resolution.py` línea 19
   
   La definición es idéntica en las 3, pero esta triplicación es exactamente el anti-patrón que queríamos eliminar.

5. **El `__init__.py` re-exporta funciones que nadie importa externamente:**
   El archivo `steps/__init__.py` tiene un `__all__` completo pero nadie importa desde `src.graph.steps` directamente — graph.py importa directamente desde los sub-módulos.

**Veredicto:** 🔴 **Incompleto y con riesgo alto.** Claude Code creó los módulos correctamente (la estructura y el código extraído son buenos), pero no completó el trabajo:
- No eliminó las funciones originales de graph.py.
- No actualizó las llamadas en graph.py para usar los imports.
- graph.py tiene ahora más código que antes (114 líneas más).
- El resultado es código duplicado, no código extraído.

**¿Por qué los tests pasan?** Porque graph.py sigue usando sus propias definiciones locales. Los imports muertos no causan errores, solo código innecesario. Los módulos en `steps/` son copias funcionales pero no se usan.

**Riesgo:** 🔴 **Alto.** Dos copias de cada función crea un riesgo de divergencia — si alguien modifica una versión y no la otra, el comportamiento será inconsistente.

---

## 🔍 HALLAZGOS TRANSVERSALES

### 1. Encoding de comentarios
Los comentarios y banners insertados por Claude Code muestran caracteres Unicode mal renderizados en unicode:
```
# ── Token sets removed (seniority refactoring) ──────────
```
Se muestra como `ÔöÇÔöÇ` en el diff, lo que sugiere un problema de encoding. Esto es cosmético pero indica que Claude Code usó caracteres box-drawing UTF-8 que el terminal/git renderizó con encoding incorrecto.

### 2. Patrón de "movimiento lateral" en token sets
En las Fases 1, 3, y 4, el patrón dominante es: eliminar listas de tokens como constantes globales → reintroducirlas como tuplas/sets literales inline o locales. Esto es una mejora de encapsulación pero NO es una eliminación real de hardcoding. La diferencia es que:
- Antes: ~120 tokens en constantes globales visibles y crecientes
- Ahora: ~30 tokens en líneas inline menos visibles pero aún estáticas

### 3. Calidad de la extracción de módulos
Los 3 archivos en `steps/` están **bien escritos**:
- Docstrings claros
- Tipos correctos
- Imports mínimos y correctos
- La lógica es idéntica a la original en graph.py

El problema NO es la calidad del código extraído, sino que la extracción está incompleta (las originales no se borraron y los call sites no se actualizaron).

---

## 📊 SCORECARD FINAL

| Categoría | Puntuación |
|---|---|
| **Correctitud funcional** | 9/10 — Tests pasan, no se rompió nada |
| **Completitud vs la auditoría** | 7/10 — 8 de 12 fases completamente correctas |
| **Calidad de código nuevo** | 8/10 — Bien escrito, buena estructura |
| **Eliminación real de hardcoding** | 6/10 — Muchos tokens se reorganizaron, no se eliminaron |
| **graph.py desacoplamiento** | 3/10 — Módulos creados pero funciones duplicadas y no usadas |
| **Seguridad del refactoring** | 8/10 — Conservador, no rompió tests |

---

## ⚡ ACCIONES PENDIENTES (ordenadas por prioridad)

### P0 — Críticas

1. **Completar la extracción de graph.py:**
   - Eliminar las funciones originales de graph.py que ya están en `steps/`
   - Actualizar las llamadas en graph.py para usar los imports de `steps/`
   - O bien: eliminar los imports alias `_steps_*` y usar los nombres originales directamente
   - Eliminar las 2 copias extras de `_norm_name` (mantener solo 1 en un lugar compartido)

2. **Restaurar security_sandbox como gate incondicional en QA:**
   - `resolve_qa_gates()` debería mantener al menos `security_sandbox` como gate HARD incluso cuando no hay gates en el contrato. La eliminación de `CONTRACT_BROKEN_FALLBACK_GATES` removió un gate de seguridad que debería ser incondicional.

### P1 — Importantes

3. **Eliminar la duplicación de CAPABILITY_DETECTION_PROMPT** en execution_planner.py (aparece 2 veces).

4. **Considerar mover los tokens inline de `_strategy_mentions_resampling` y `_build_visual_requirements` a queries al contrato LLM-generado**, ya que el CAPABILITY_DETECTION_PROMPT ya le pide al LLM que setee estos flags.

### P2 — Cosméticas

5. **Arreglar encoding de caracteres** en los comentarios banner (caracteres box-drawing → ASCII simple `---` o `===`).

6. **Limpiar los archivos diff temporales** generados durante esta auditoría.

---

## 🏆 CONCLUSIÓN

Claude Code realizó un trabajo **sólido y conservador** — priorizó no romper tests por encima de completitud. Las **8 fases simples** (F2, F6, F7, F8, F9, F10, F12, tests) se implementaron correctamente y con buen criterio. Las **3 fases de eliminación de tokens** (F1, F3, F4) se ejecutaron con una estrategia de "reorganizar y reducir" en vez de "eliminar completamente", lo cual es un compromiso razonable para un refactoring.

El **problema principal** es la **Fase 11 (graph.py):** los módulos de steps/ se crearon correctamente pero su integración está incompleta — las funciones están duplicadas y los imports no se usan. Esto necesita completarse para que el refactoring tenga efecto real sobre graph.py.

**Score del trabajo de Claude Code: 7.5/10**

Para llegar a 10/10: completar la extracción de graph.py y restaurar el gate `security_sandbox` incondicional.
