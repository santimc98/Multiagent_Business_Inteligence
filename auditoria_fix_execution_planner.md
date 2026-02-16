# Auditoría: run_6ca09c47 — Fallo del Execution Planner

**Run ID:** `6ca09c47`
**Fecha:** 2026-02-16 19:57–20:03 UTC (6 min 28 s)
**Modelo:** `gemini-3-flash-preview`
**Status final:** `REJECTED` → `execution_contract_invalid`

---

## Veredicto

**FALLO ARQUITECTÓNICO**, no fallo de razonamiento del agente.

El agente (Gemini 3 Flash) produjo contratos semánticamente correctos cuando tuvo tokens suficientes. La arquitectura lo estranguló con un presupuesto de tokens de salida insuficiente (`max_output_tokens=4000`) que nunca se escala en reintentos, combinado con gates de calidad demasiado estrictos para los casos marginales, creando una espiral de muerte irrecuperable.

---

## Cronología de los 7 intentos

| # | Modo | finish_reason | Output tokens | JSON válido? | Quality? | Problema |
|---|------|---------------|---------------|-------------|----------|----------|
| 1 | progressive r1 | **1 (STOP)** | 1,691 | **Sí** | **Error** (2E, 1W) | Parcial; falla `outcome_columns_sanity` + `llm_min_contract_divergence` |
| 2 | progressive r2 | **2 (MAX_TOKENS)** | 1,144 | No | — | JSON truncado: `'[' was never closed` |
| 3 | sectional core r1 | **2 (MAX_TOKENS)** | 1,123 | No | — | JSON truncado: `'{' was never closed` |
| 4 | sectional core r2 | **2 (MAX_TOKENS)** | 148 | No | — | JSON truncado: `'{' was never closed` (apenas 613 chars) |
| 5 | quality r1 | **1 (STOP)** | 2,225 | **Sí** | **Error** (2E, 1W) | `outcome_columns_sanity` + `iteration_policy_limits` + `llm_min_contract_divergence` |
| 6 | quality r2 | **2 (MAX_TOKENS)** | 143 | No | — | JSON truncado (563 chars) |
| 7 | quality r3 | **2 (MAX_TOKENS)** | 143 | No | — | JSON truncado (563 chars) |

**Patrón claro:** 5 de 7 intentos mueren por MAX_TOKENS. Los 2 que completan JSON son rechazados por quality gates.

---

## Causa Raíz #1: Token budget insuficiente y estático

**Archivo:** [execution_planner.py:4687-4704](src/agents/execution_planner.py#L4687-L4704)

```python
max_output_tokens = 4000  # hardcoded
self._default_max_output_tokens = max(4000, max_output_tokens)
```

El contrato v4.1 completo (como el de `response_attempt_1.txt`) tiene ~287 líneas de JSON denso. Con `max_output_tokens=4000`, el modelo genera ~1100-2200 tokens antes de agotar el presupuesto.

**Agravante:** La función `_generation_config_for_prompt` (línea 4742) solo puede **reducir** el límite, nunca incrementarlo:

```python
budgeted_max = min(self._default_max_output_tokens, max(1024, int(available)))
```

Cuando el prompt de repair es más largo (72K chars vs 58K chars inicial), el presupuesto disponible **se reduce aún más**, creando una espiral descendente.

### Evidencia directa

- Intento 4: prompt 38K chars → solo 148 tokens de output (613 chars)
- Intentos 6-7: prompt 38K chars → solo 143 tokens de output (563 chars)
- El modelo literalmente no puede completar ni la primera sección del contrato

---

## Causa Raíz #2: No hay escalación de token budget en retries

La lógica de retry escala el **modo** de compilación (progressive → sectional → quality), pero **nunca escala `max_output_tokens`**. Cada reintento usa el mismo techo de 4000 tokens.

Los prompts de repair son **más largos** que los originales porque incluyen:
- El contrato anterior completo
- Los errores de validación
- Las instrucciones de reparación

Esto crea un **efecto contraproducente**: más contexto de entrada → menos presupuesto de salida → más truncamiento → más errores.

---

## Causa Raíz #3: Quality gates demasiado estrictos en el ciclo de reparación

Los 2 intentos que SÍ produjeron JSON completo (intentos 1 y 5) generaron contratos **semánticamente correctos**:

- `column_roles.outcome: ["target"]` — correcto
- `allowed_feature_sets.model_features`: 13 features clínicas — correcto
- `forbidden_features: ["id", "target", "is_train"]` — correcto
- `validation_requirements: stratified_kfold, k=5, roc_auc` — correcto
- `cleaning_gates` y `qa_gates` — razonables

Pero fueron rechazados por:

1. **`contract.outcome_columns_sanity`** (error): El validator compara `_collect_outcome_columns()` contra `_collect_target_candidates()`. Dado que el contrato tiene `outcome: ["target"]` y steward_semantics tiene `primary_target: "target"`, la lógica debería pasar. Posible causa: discrepancia sutil en normalización o un campo adicional no presente en el contrato que el validator espera.

2. **`contract.llm_min_contract_divergence`** (error/warning): Divergencia semántica entre el contrato LLM y el scaffold determinístico. El planner compara campos clave y si la divergencia Jaccard supera un umbral, rechaza. El contrato del LLM es razonable pero puede diferir en detalles menores del scaffold (p.ej., `is_train` en `pre_decision` vs scaffold que podría no incluirlo ahí).

3. **`contract.iteration_policy_limits`** (warning): Usa keys no-canónicas (`max_cleaning_retries` vs lo que el validator espera).

**Impacto:** Estos rechazos fuerzan reparaciones que, con el token budget ya agotado, garantizan el fracaso.

---

## Contrato producido vs contrato necesario

El contrato del intento 5 (`response_attempt_1.txt`) era **funcional y ejecutable**:

- Scope: `full_pipeline`
- Column roles correctamente mapeados
- Data types razonables
- Cleaning gates y QA gates definidos
- Runbooks para data_engineer y ml_engineer
- submission.csv con formato correcto (`id`, `Heart Disease`)

**Este contrato debería haber sido aceptado**, posiblemente con warnings en lugar de errors para las divergencias menores.

---

## Recomendaciones de Fix

### P0 — Incrementar y escalar token budget

```python
# En __init__:
self._default_max_output_tokens = max(8192, max_output_tokens)

# En _generation_config_for_prompt, permitir escalación:
# Añadir parámetro min_output_tokens para retries
def _generation_config_for_prompt(self, prompt, min_output_tokens=None):
    ...
    floor = min_output_tokens or 1024
    budgeted_max = min(self._default_max_output_tokens, max(floor, int(available)))
```

O mejor: usar env var `EXECUTION_PLANNER_MAX_OUTPUT_TOKENS=8192` como fix inmediato.

### P1 — Escalar token budget en cada retry

En la lógica de retry, incrementar `max_output_tokens` progresivamente:
- Intento 1: 8192
- Intento 2: 12288
- Intento 3+: 16384

### P2 — Relajar quality gates para contratos casi-válidos

Cambiar `outcome_columns_sanity` a **warning** (no error) cuando la divergencia es mínima y el outcome contiene solo columnas plausibles de target.

Evaluar si `llm_min_contract_divergence` debería bloquear la aceptación o solo advertir cuando el contrato es estructuralmente completo.

### P3 — Fallback al mejor contrato parcial

Si todos los intentos fallan, en lugar de emitir `{}`, usar el **mejor contrato que pasó JSON parsing** (intentos 1 o 5) con warnings explícitos. Un contrato con 2 warnings es infinitamente mejor que un contrato vacío que mata toda la pipeline.

---

## Resumen ejecutivo

| Categoría | Veredicto |
|-----------|-----------|
| **Tipo de fallo** | Arquitectónico |
| **Razonamiento del agente** | Correcto — produjo contratos válidos cuando tuvo presupuesto |
| **Causa primaria** | `max_output_tokens=4000` insuficiente para contrato v4.1 |
| **Causa secundaria** | Sin escalación de tokens en retries |
| **Causa terciaria** | Quality gates demasiado estrictos rechazan contratos funcionales |
| **Fix inmediato** | `EXECUTION_PLANNER_MAX_OUTPUT_TOKENS=8192` en `.env` |
| **Fix estructural** | Escalación progresiva de tokens + fallback al mejor contrato parcial |
