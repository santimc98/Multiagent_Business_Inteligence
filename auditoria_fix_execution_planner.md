# Auditoría Forense — Run 2089eda7
## Heart Disease Binary Classification (900K rows, 16 cols)

**Run ID**: `2089eda7`
**Dataset**: `business_input.csv` — 900,000 rows × 16 columnas (todas typed como `object`)
**Objetivo**: Clasificación binaria — predecir `target` (Heart Disease), maximizar ROC-AUC con Stratified K-Fold sobre `is_train=1`, generar `submission.csv` con `[id, Heart Disease]` probabilidades.
**Resultado**: `NO_GO` — pipeline abortado, 0 métricas, 0 predicciones.

---

## 1. Resumen Ejecutivo

La run falló por **2 causas raíz independientes** que se refuerzan mutuamente:

| # | Causa Raíz | Tipo | Agente Responsable | Impacto |
|---|-----------|------|-------------------|---------|
| **RC-1** | `trim_to_budget()` recibe keyword arg que no acepta | Bug de código | ml_engineer.py ↔ contract_views.py | Hard crash: ML engineer no genera código, ML plan vacío |
| **RC-2** | Planner clasifica 7 features como "outcome" y elimina 6 features | Error de razonamiento LLM | Execution Planner | Contrato envenenado: modelo tendría 0 features útiles |

**Ambas causas deben corregirse.** Si solo se corrige RC-1, el ML engineer ejecutaría con `model_features: ["is_train"]` y `forbidden_features: [todas las features reales]` → modelo basura. Si solo se corrige RC-2, el crash de código impide cualquier ejecución.

---

## 2. Timeline de la Run

```
18:03:xx  Steward          ✅  Análisis correcto: target="target", 13 features, id, is_train
18:04:xx  Strategist       ✅  Strategy "predictive", ROC-AUC, K-Fold, CatBoost fallback
18:05:xx  Domain Expert    ✅  Critique generada
18:06:xx  Execution Planner ⚠️  Contrato ACEPTADO (llm_success=true) pero con errores lógicos graves
18:07:xx  Data Engineer    ✅  Código ejecuta en 8.42s, exit_code=0 — pero sigue contrato envenenado
18:07:xx  Cleaning Reviewer ✅  APPROVED — valida compliance, no detecta contrato incorrecto
18:08:xx  ML Plan          ❌  CRASH: trim_to_budget() → plan vacío (source="render_error")
18:08:xx  ML Engineer      ❌  CRASH: mismo error → ml_engineer_host_crash.txt
18:08:xx  → route "failed" → Translator (fallback determinístico)
18:09:xx  Run finalize: NO_GO — missing scored_rows.csv, metrics.json, alignment_check.json
```

---

## 3. Evaluación por Agente

### 3.1 Steward — ✅ Excelente (95/100)

El Steward identificó correctamente:

- `target` como variable dependiente binaria (1=Presence, 0=Absence)
- `is_train` como columna de partición (70/30 split)
- `id` como identificador único
- 13 features clínicas (6 continuas + 7 categóricas)
- Todas las columnas tipadas como "object" (requieren conversión numérica)
- Target nulls (30%) alineados con `is_train=0`

**dataset_semantics.json** — impecable:
```json
{
  "primary_target": "target",
  "split_candidates": ["is_train"],
  "id_candidates": ["id"],
  "notes": ["Categorical variables should be treated as discrete features."]
}
```

**column_manifest** — correcto:
```
schema_mode: normal | total_columns: 16
anchors: [id, target, is_train]
families: [PREDICTOR_FEATURES(all_columns_except, count=13, role=feature)]
```

**Señal clave que downstream ignoró**: El Steward dice explícitamente "*Categorical variables are provided as numeric encodings and should be treated as discrete features*". Esta información llegó al Planner pero fue ignorada.

### 3.2 Strategist — ✅ Bueno (80/100)

- `objective_type: "predictive"` ✅
- Métricas: ROC-AUC, LogLoss, PR-AUC ✅
- Validación: Stratified K-Fold (K=5) ✅
- Fallback chain: CatBoost + LightGBM → Single CatBoost → Logistic Regression ✅
- Feasibility analysis: correcto (486K train rows, 13 features, high power) ✅

**Nota**: El strategy spec no incluye lista explícita de `target_columns` ni `feature_columns`. Esto deja al Planner con libertad excesiva para interpretar roles.

### 3.3 Execution Planner — ❌ Fallo Crítico (25/100)

**El contrato fue aceptado por el validador** (`llm_success: true`, 0 errors, 1 warning). Esto confirma que los fixes del P0 anterior funcionaron: `column_dtype_targets` ahora tiene el formato correcto con `target_dtype`.

**Sin embargo, el contrato tiene errores lógicos catastróficos** que el validador no detecta:

#### Error 1: 6 features eliminadas de `canonical_columns`

| Columna eliminada | Tipo | Importancia clínica |
|-------------------|------|-------------------|
| `age` | Continua | Factor de riesgo primario |
| `bp` | Continua | Presión arterial |
| `cholesterol` | Continua | Factor lipídico clave |
| `max_hr` | Continua | Frecuencia cardíaca máxima |
| `st_depression` | Continua | Hallazgo EKG (alta señal) |
| `thallium` | Categórica (3 vals) | Test de perfusión (muy predictiva) |

El dataset tiene 16 columnas. El contrato solo incluye 10 en `canonical_columns`. Las 6 columnas continuas más importantes para predicción cardíaca fueron eliminadas silenciosamente.

#### Error 2: 7 features clasificadas como "outcome"

```json
"column_roles": {
  "pre_decision": ["is_train"],           // ← única "feature" → inútil
  "outcome": [
    "target",              // ← CORRECTO: es el target
    "sex",                 // ← INCORRECTO: es feature
    "chest_pain_type",     // ← INCORRECTO: es feature
    "fbs_over_120",        // ← INCORRECTO: es feature
    "ekg_results",         // ← INCORRECTO: es feature
    "exercise_angina",     // ← INCORRECTO: es feature
    "slope_of_st",         // ← INCORRECTO: es feature
    "num_vessels_fluro"    // ← INCORRECTO: es feature
  ],
  "identifiers": ["id"]
}
```

**Consecuencia directa** — `allowed_feature_sets` se derivó automáticamente:
```json
"model_features":     ["is_train"],     // ← solo el split column
"forbidden_features": ["target", "sex", "chest_pain_type", ...]  // ← TODAS las features reales
```

#### Error 3: `evaluation_spec` vacío

```json
"evaluation_spec": { "objective_type": "predictive" }
```

Sin metric, sin CV strategy, sin split column. El ML engineer no sabría qué optimizar.

#### Análisis de por qué el LLM falló

El Planner recibió toda la información correcta:
- Steward summary con `primary_target: "target"` y features correctamente categorizadas
- Column manifest con `PREDICTOR_FEATURES(count=13, role=feature)`
- Strategy spec con ROC-AUC y K-Fold

**Patrón del error**: El LLM trató toda columna categórica/binaria como "outcome" (confusión semántica entre "resultado de observación médica" y "variable objetivo a predecir"). Las 6 columnas continuas no se incluyeron en ninguna categoría.

**Causa del prompt**: La especificación del contrato dice únicamente:
```
- column_roles: object mapping role -> list[str]
```

Sin definición de qué significa cada rol:
- `outcome` = **solo** la(s) variable(s) target que el modelo predice
- `pre_decision` = features disponibles para el modelo
- `decision` = la predicción del modelo

Esta ambigüedad permite interpretaciones incorrectas. En un contexto médico, "sex" y "chest_pain_type" *son* "outcomes de observación" — pero en ML, son *features*.

### 3.4 Data Engineer — ✅ Correcto pero envenenado (N/A)

El DE generó código correcto que sigue el contrato al pie de la letra:
```python
REQUIRED_COLUMNS = [
    "id", "target", "is_train", "sex", "chest_pain_type",
    "fbs_over_120", "ekg_results", "exercise_angina",
    "slope_of_st", "num_vessels_fluro"
]
df = df[REQUIRED_COLUMNS].copy()  # ← 6 features eliminadas
```

- Ejecutó en 8.42 segundos ✅
- 900,000 rows → 900,000 rows (0% drop) ✅
- Conversiones de tipo correctas (object → float64/int64) ✅
- Gates: `required_columns_present` ✅, `id_integrity` ✅, `no_semantic_rescale` ✅

**El DE no tiene culpa.** Su trabajo es ejecutar el contrato, no cuestionar si el contrato es correcto. El contrato dice 10 columnas, el DE produce 10 columnas.

### 3.5 Cleaning Reviewer — ⚠️ Aprobó contrato envenenado (50/100)

```json
{"status": "APPROVED", "feedback": "all gates passed"}
```

El reviewer verificó:
- ✅ `required_columns_present` — las 10 columnas del contrato están presentes
- ✅ `id_integrity` — id sin notación científica
- ✅ `no_semantic_rescale` — sin símbolos de porcentaje

**Deficiencia arquitectónica**: El reviewer valida *compliance con el contrato*, no *correctness del contrato*. No tiene un gate que compare las columnas del contrato contra el data atlas o steward output. Si el contrato dice "solo 10 columnas", el reviewer aprueba 10 columnas sin cuestionar.

### 3.6 ML Plan — ❌ Crash (0/100)

```json
{
  "plan_source": "render_error",
  "primary_metric": "unspecified",
  "cv_policy": {"strategy": "unspecified"},
  "notes": ["Failed to render prompt context: trim_to_budget() got an unexpected keyword argument 'max_str_len'"]
}
```

El ML plan se genera antes del ML engineer code. Cuando `_serialize_json_for_prompt()` intentó comprimir el contexto para el prompt, llamó a `trim_to_budget()` con `max_str_len=700` — argumento que no existe en la función. El plan cayó a defaults vacíos.

### 3.7 ML Engineer — ❌ Hard Crash (0/100)

```
TypeError: trim_to_budget() got an unexpected keyword argument 'max_str_len'
  at ml_engineer.py:807 → _serialize_json_for_prompt
  at ml_engineer.py:2258 → generate_code
  at graph.py:16184 → run_engineer
```

El crash ocurre **antes de generar código**. El ML engineer nunca llegó a llamar al LLM. Los 23 sitios en `ml_engineer.py` que pasan `max_str_len` a `trim_to_budget` están todos rotos.

**Post-crash routing**: `graph.py` → `check_engineer_success` → `"failed"` → `"translator"`. No hay retry para crashes de código (solo para fallos de sandbox).

### 3.8 Translator — ⚠️ Fallback determinístico (40/100)

```markdown
# Reporte Ejecutivo (Fallback Determinístico)
## Decisión Ejecutiva: GO_WITH_LIMITATIONS
```

El translator también habría hit el `trim_to_budget` bug al serializar contexto, así que cayó al fallback determinístico. El reporte dice "GO_WITH_LIMITATIONS" cuando el `run_summary` dice "NO_GO" — inconsistencia notada en el reporte (`Decision Reconciliation Note`).

---

## 4. Root Cause Analysis

### RC-1: Bug de código — `trim_to_budget()` signature mismatch

**Definición de la función** (`contract_views.py:1037`):
```python
def trim_to_budget(obj: Any, max_chars: int) -> Any:
    max_str_len = 1200       # ← variable INTERNA
    max_list_items = 25      # ← variable INTERNA
    for _ in range(4):
        trimmed = _trim_value(obj, max_str_len, max_list_items, ...)
        if len(json.dumps(trimmed)) <= max_chars:
            return trimmed
        max_str_len = max(200, int(max_str_len * 0.7))
        max_list_items = max(8, int(max_list_items * 0.7))
```

**Llamada del ml_engineer** (`ml_engineer.py:807`):
```python
trimmed = trim_to_budget(
    compact,
    max_chars=max_chars,      # ✅ acepta
    max_str_len=max_str_len,  # ❌ TypeError: unexpected keyword
    max_list_items=max_list_items,  # ❌ TypeError
)
```

**23 call sites afectados** en ml_engineer.py: líneas 810, 1555, 1561, 1567, 2261, 2272, 2303, 2309, 2315, 2324, 2366, 2378, 2398, 2404, 2410, 2416, 2422, 2433, 2444, 2450, 2456, 2465.

**Alcance del impacto**: Solo `ml_engineer.py` importa `trim_to_budget` directamente. Los otros consumidores llaman la función internamente dentro de `contract_views.py` donde siempre se usa con la signature correcta de 2 args.

**Fix** (opción A — recomendada, 5 minutos):

Actualizar `trim_to_budget` para aceptar los kwargs:
```python
def trim_to_budget(
    obj: Any,
    max_chars: int,
    max_str_len: int = 1200,
    max_list_items: int = 25,
) -> Any:
```

**Fix** (opción B — más invasiva):
Actualizar los 23 call sites en ml_engineer.py para pasar solo `max_chars`.

### RC-2: Planner column role misclassification

**Información recibida por el Planner** (correcta):

| Fuente | Dato | Correcto |
|--------|------|----------|
| Steward | `primary_target: "target"` | ✅ |
| Steward | "Categorical variables should be treated as discrete features" | ✅ |
| Column manifest | `PREDICTOR_FEATURES(count=13, role=feature)` | ✅ |
| Column inventory | 16 columnas listadas | ✅ |
| Strategy | ROC-AUC, K-Fold, CatBoost | ✅ |

**Contrato generado por el LLM** (incorrecto):

| Campo | Valor generado | Valor correcto |
|-------|---------------|----------------|
| `canonical_columns` | 10 columnas | 16 columnas |
| `outcome_columns` | 8 columnas | 1 (`target`) |
| `model_features` | `["is_train"]` | 13 features |
| `forbidden_features` | 8 columnas (todas las features) | `["target"]` |

**¿Por qué `build_contract_min()` no salvó la situación?**

La función determinística `build_contract_min()` habría generado roles correctos:
```python
# build_contract_min (línea ~2780)
outcome_cols = _resolve_candidate_targets()  # → ["target"] solamente
pre_decision = [col for col in canonical_columns if col not in assigned]  # → 13 features
```

Pero `build_contract_min()` no se usa como validación/comparación contra el contrato LLM. El contrato LLM fue aceptado (0 errores de formato) y se persistió tal cual.

### RC-3 (Arquitectónico): Validación de correctness ausente

El sistema tiene validación de **formato** pero no de **semántica**:

| Qué se valida | ¿Existe? | Ejemplo |
|---------------|----------|---------|
| column_dtype_targets tiene "target_dtype" | ✅ | (fix anterior P0) |
| canonical_columns es lista de strings | ✅ | |
| column_roles es dict con listas | ✅ | |
| canonical_columns cubre todas las columnas del inventario | ❌ | 6 features faltantes |
| outcome_columns ⊆ target candidates del steward | ❌ | 7 features clasificadas como outcome |
| model_features tiene al menos 1 feature real | ❌ | Solo "is_train" |
| Contrato LLM ≈ build_contract_min (cross-check) | ❌ | Divergencia total |

---

## 5. Cascada de Daño

```
                    ┌──────────────────┐
                    │  Steward (✅)     │
                    │  16 cols, target  │
                    │  = "target" only  │
                    └────────┬─────────┘
                             │ información correcta
                    ┌────────▼─────────┐
                    │  Planner (❌)     │
                    │  Clasifica mal   │──── RC-2: 7 features → "outcome"
                    │  Elimina 6 cols  │──── RC-2: 6 features eliminadas
                    └────────┬─────────┘
                             │ contrato envenenado
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐  ┌──▼────────┐  ┌──▼────────────┐
     │  DE (✅ código  │  │ Reviewer   │  │ ML Plan (❌)  │
     │  ❌ resultado) │  │ (⚠️ aprueba│  │ trim_to_budget│── RC-1
     │  10 cols output│  │  envenen.) │  │ crash → vacío │
     └────────────────┘  └───────────┘  └──────┬────────┘
                                               │
                                        ┌──────▼────────┐
                                        │ ML Engineer   │
                                        │ (❌ crash)    │── RC-1
                                        │ 0 código      │
                                        └──────┬────────┘
                                               │ "failed"
                                        ┌──────▼────────┐
                                        │ Translator    │
                                        │ (⚠️ fallback) │
                                        │ GO_WITH_LIMIT │
                                        └───────────────┘
```

---

## 6. Fixes Priorizados

### P0 — Hotfix inmediato (10 min)

**P0-1: Fix `trim_to_budget` signature**

En `contract_views.py`, actualizar la función para aceptar los kwargs que ml_engineer ya envía:

```python
# ANTES (línea 1037)
def trim_to_budget(obj: Any, max_chars: int) -> Any:
    max_str_len = 1200
    max_list_items = 25

# DESPUÉS
def trim_to_budget(
    obj: Any,
    max_chars: int,
    max_str_len: int = 1200,
    max_list_items: int = 25,
) -> Any:
```

Esto desbloquea: ML Plan, ML Engineer, Translator. **Impacto: 100% de la run puede avanzar.**

### P1 — Fixes de contrato (30 min)

**P1-1: Añadir definiciones de roles al prompt del Planner**

En `MINIMAL_CONTRACT_COMPILER_PROMPT` (línea ~203), cambiar:
```
ANTES:
- column_roles: object mapping role -> list[str]

DESPUÉS:
- column_roles: object mapping role -> list[str]
  Role definitions (ML context):
  - "outcome": ONLY the target variable(s) that the model predicts. Usually 1 column.
    Do NOT include features here even if they are binary or categorical.
  - "pre_decision": ALL feature columns available for modeling. This includes
    numerical, categorical, binary features — anything the model can use as input.
  - "decision": The model's prediction output column(s) (usually empty in contract).
  - "identifiers": ID/key columns (no predictive value).
  - "post_decision_audit_only": Columns used only for post-hoc analysis.

  CRITICAL: In a predictive task, outcome = ONLY the target. All other columns
  (sex, age, chest_pain_type, etc.) are pre_decision features, NOT outcomes.
```

**P1-2: Añadir validación de canonical_columns coverage**

En `contract_validator.py`, agregar regla:
```python
# Verificar que canonical_columns cubre al menos 80% del inventory
inventory_set = set(column_inventory or [])
canonical_set = set(contract.get("canonical_columns", []))
coverage = len(canonical_set & inventory_set) / max(len(inventory_set), 1)
if coverage < 0.8:
    issues.append({
        "rule": "contract.canonical_columns_coverage",
        "severity": "error",
        "message": f"canonical_columns covers only {coverage:.0%} of column inventory. "
                   f"Missing: {sorted(inventory_set - canonical_set)[:10]}"
    })
```

**P1-3: Validar que outcome_columns ⊆ steward target candidates**

```python
steward_targets = set(steward_semantics.get("primary_target", []))
# ... add split_candidates, id_candidates
outcome_set = set(contract.get("outcome_columns", []))
unexpected_outcomes = outcome_set - steward_targets - {"target"}
if unexpected_outcomes:
    issues.append({
        "rule": "contract.outcome_columns_sanity",
        "severity": "error",
        "message": f"outcome_columns contains non-target columns: {sorted(unexpected_outcomes)}. "
                   f"These are likely features, not outcomes."
    })
```

### P2 — Cross-validation determinístico (2h)

**P2-1: Comparar contrato LLM vs `build_contract_min()`**

Después de que el LLM genera el contrato, ejecutar `build_contract_min()` y comparar campos clave:
```python
llm_contract = ... # contrato del LLM
min_contract = build_contract_min(...)

# Comparar roles
llm_outcomes = set(llm_contract.get("outcome_columns", []))
min_outcomes = set(min_contract.get("outcome_columns", []))

if llm_outcomes != min_outcomes:
    divergence = {
        "llm_only": sorted(llm_outcomes - min_outcomes),
        "min_only": sorted(min_outcomes - llm_outcomes),
    }
    # Si la divergencia es grande (>3 columnas), rechazar contrato LLM
    if len(divergence["llm_only"]) > 3:
        # Use build_contract_min roles as authoritative
        contract["outcome_columns"] = min_contract["outcome_columns"]
        contract["column_roles"] = min_contract["column_roles"]
```

**P2-2: Gate de sanidad en Cleaning Reviewer**

Añadir gate `feature_coverage_sanity`:
```python
{
    "name": "feature_coverage_sanity",
    "severity": "SOFT",
    "params": {
        "min_feature_count": 3,
        "check_against": "data_atlas"
    }
}
```
El gate verifica que el cleaned_data.csv tiene al menos N features (excluyendo id, target, split columns). Si el dataset original tiene 13 features y el limpiado tiene 0 features, algo está muy mal.

### P3 — Robustez estructural (1 día)

**P3-1: model_features mínimo gate**

Validación que `model_features` tenga al menos 1 columna que no sea split/id:
```python
model_feats = set(contract.get("allowed_feature_sets", {}).get("model_features", []))
structural_cols = set(steward_semantics.get("split_candidates", [])) | set(steward_semantics.get("id_candidates", []))
useful_feats = model_feats - structural_cols
if not useful_feats:
    issues.append({
        "rule": "contract.model_features_empty",
        "severity": "error",
        "message": "model_features contains only structural columns (split/id). No useful features for modeling."
    })
```

**P3-2: Retry para crashes de código (no solo de sandbox)**

Actualmente, un host crash en ML engineer → route directo a translator. Agregar retry:
```python
except TypeError as e:
    if ml_attempt < max_attempts:
        # Log the crash, fix the call if possible, retry
        ...
    else:
        # Persist crash and route to translator
```

**P3-3: Strategy spec debe incluir target_columns explícitamente**

El Strategist debería emitir `target_columns: ["target"]` en su spec. Esto daría al Planner una señal redundante que podría cruzar contra la steward info.

---

## 7. Análisis Comparativo: Run 6a51bf4f vs 2089eda7

| Dimensión | Run 6a51bf4f (anterior) | Run 2089eda7 (actual) |
|-----------|------------------------|----------------------|
| Steward | ✅ | ✅ |
| Strategy | ✅ | ✅ |
| Planner formato | ❌ `"type"` vs `"target_dtype"` | ✅ (fix P0 funcionó) |
| Planner lógica | No evaluable (crash formato) | ❌ column roles incorrectos |
| Data Engineer | No ejecutó | ✅ código, ❌ resultado (envenenado) |
| ML Engineer | No ejecutó | ❌ crash `trim_to_budget` |
| Duración | 32 min (7 intentos LLM) | ~6 min |
| Tokens gastados | ~450K | ~150K |
| Resultado | `{}` contrato vacío | Contrato aceptado pero envenenado |

**Progreso**: El fix P0 anterior eliminó el bloqueo de formato. El Planner ahora produce contratos válidos sintácticamente. Pero la validación semántica sigue ausente — el contrato pasa todas las gates de formato pero tiene errores lógicos que destruyen la run.

---

## 8. Respuesta a la Pregunta Central

> ¿Es un fallo arquitectónico del sistema o un mal razonamiento de los agentes que intoxicó la run?

**Es ambos, pero con clara jerarquía:**

**1. El error primario es del Planner LLM** (razonamiento). El LLM tuvo toda la información correcta (steward, manifest, strategy) y generó un contrato con column roles catastróficamente incorrectos. Es un error de razonamiento del LLM: confundió "outcome" médico con "outcome" de ML.

**2. El sistema amplifica el error en lugar de contenerlo** (arquitectura). No hay ningún check que diga "el contrato dice 0 features útiles, eso no puede estar bien". El cleaning reviewer aprueba. La integrity audit pasa. El run_facts_pack muestra `target_columns: []`. Nadie levanta la alarma.

**3. El bug de código es independiente y bloquea toda la pipeline** (`trim_to_budget` signature). Incluso un contrato perfecto habría crasheado en ML engineer.

**Metáfora**: El Planner escribió una receta que dice "usar solo agua como ingrediente para hacer un pastel". El Reviewer verificó que el agua es potable y aprobó. El Chef intentó encender el horno pero se electrocutó (bug de código) antes de descubrir que la receta es imposible.

---

## 9. Costo de la Run

| Recurso | Valor | Desperdiciado |
|---------|-------|---------------|
| Tiempo total | ~6 min | 100% |
| LLM calls (steward) | 1 | ✅ pero resultado ignorado |
| LLM calls (strategy) | 1 | ✅ pero resultado ignorado |
| LLM calls (planner) | 1 | Contrato con errores lógicos |
| LLM calls (DE) | 1 | Código correcto, resultado inútil |
| DE sandbox | 8.42s | 900K rows procesados para nada |
| ML calls | 0 (crash) | — |

**Con P0 (10 min de desarrollo)**: La run habría llegado a ML engineer sin crash.
**Con P0+P1 (40 min de desarrollo)**: La run habría usado las 13 features correctas, generado modelo y predicciones.