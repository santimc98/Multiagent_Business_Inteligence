# Auditoría: run_408f3dc7 — Fallo del ML Engineer

**Run ID:** `408f3dc7`
**Fecha:** 2026-02-16 21:38–22:24 UTC (45 min)
**Modelo ML Engineer:** `moonshotai/kimi-k2.5`
**Status final:** `NEEDS_IMPROVEMENT` / `REJECTED`
**CV ROC-AUC alcanzado:** 0.9138 (+/- 0.0009) — excelente rendimiento

---

## Veredicto

**Tres causas raíz encadenadas, ninguna es el razonamiento del ml_engineer:**

1. **BUG en artifact content validator** (causa primaria) — Falso positivo en `scored_rows_unknown_columns`
2. **Contrato con forbidden_features incorrecto** (causa del execution_planner) — 7 features clínicas prohibidas erróneamente
3. **Bug menor del agente** — `log_loss(eps=)` API deprecada (intento 1 solamente)

---

## Cronología de ejecuciones

| Intento | Exit Code | Duración | Resultado real | Lo que el sistema creyó |
|---------|-----------|----------|----------------|------------------------|
| 1 | **1** | 42.8s | FAIL: `log_loss(eps=1e-15)` TypeError | Correcto: runtime error |
| 2 | **0** | 174.2s | **OK** — todos los artifacts generados | **FALSO**: runtime error |
| 3 | **0** | 213.8s | **OK** — todos los artifacts generados | **FALSO**: runtime error |
| 4 | **0** | 171.7s | **OK** — AUC 0.9138 | **FALSO**: runtime error |

**Los intentos 2, 3 y 4 ejecutaron exitosamente** pero el sistema los marcó como fallidos.

---

## Causa Raíz #1: Falso positivo en `_validate_artifact_content`

**Archivo:** `src/graph/graph.py`, líneas 6950-7075

### Cadena causal completa

1. El script genera `scored_rows.csv` con columna **"Heart Disease"** (nombre requerido por el business objective)
2. La función `_validate_artifact_content` verifica las columnas de scored_rows contra una lista de "allowed columns"
3. **"Heart Disease"** no está en la lista porque:
   - No es una columna del cleaned_data.csv (que tiene "target", no "Heart Disease")
   - No viene de `get_derived_column_names()` (no es un feature derivado)
   - El contrato define `scored_rows_schema.required_columns: ["id", "Heart Disease"]` PERO el código lee de `artifact_requirements.file_schemas` (línea 6955), que es una **key diferente**
   - `_norm_name("Heart Disease")` → `"heartdisease"` que no matchea ningún token fuzzy (`pred`, `score`, `prob`, etc.)
4. Se emite `scored_rows_unknown_columns:Heart Disease`
5. Se inyecta `EXECUTION ERROR: ARTIFACT_CONTENT_INVALID: scored_rows_unknown_columns:Heart Disease` en `execution_output`
6. `check_execution_status()` detecta `"EXECUTION ERROR"` en output → `has_error=True`
7. Se redirige a `retry_fix` → quema otro intento
8. Tras 3 retries → `finalize_runtime_failure()` → `runtime_fix_terminal=True`
9. Todos los reviewers reciben `runtime_status=FAILED_RUNTIME` → REJECTED automático

### El bug específico

**Línea 6954-6957:**
```python
reqs = contract.get("artifact_requirements") or {}
schema = _normalize_schema(reqs.get("file_schemas"))  # ← Lee "file_schemas"
scored_schema = schema.get("data/scored_rows.csv")
```

Pero el contrato almacena el schema en `artifact_requirements.scored_rows_schema`, no en `artifact_requirements.file_schemas.data/scored_rows.csv`. Las `required_columns: ["id", "Heart Disease"]` del scored_rows_schema **nunca se leen**.

### Fix necesario

En la función que construye `allowed_cols` (línea ~6960-6968), añadir lectura de `scored_rows_schema.required_columns`:

```python
# Además de file_schemas, leer scored_rows_schema del contrato
scored_rows_schema_direct = reqs.get("scored_rows_schema") or {}
if isinstance(scored_rows_schema_direct, dict):
    sr_required = scored_rows_schema_direct.get("required_columns")
    if isinstance(sr_required, list):
        allowed_cols.extend([str(col) for col in sr_required if col])
```

---

## Causa Raíz #2: Contrato con forbidden_features incorrecto

**Origen:** execution_planner (modelo gemini-3-flash-preview)

El contrato generado tiene:

```json
"model_features": ["is_train", "age", "bp", "cholesterol", "max_hr", "st_depression", "thallium"],
"forbidden_features": ["target", "sex", "chest_pain_type", "fbs_over_120", "ekg_results",
                        "exercise_angina", "slope_of_st", "num_vessels_fluro"]
```

### Problemas:

| Feature prohibida | Importancia clínica | Debería ser |
|---|---|---|
| `sex` | Alta (factor de riesgo cardiovascular) | model_feature |
| `chest_pain_type` | **Muy alta** (indicador primario) | model_feature |
| `fbs_over_120` | Moderada | model_feature |
| `ekg_results` | Alta | model_feature |
| `exercise_angina` | **Muy alta** | model_feature |
| `slope_of_st` | Alta | model_feature |
| `num_vessels_fluro` | **Muy alta** (hallazgo de cateterismo) | model_feature |
| `is_train` (en model_features) | **Ninguna** — es el split indicator | Debería ser forbidden/structural |

El planner **prohibió todas las features categóricas** y dejó solo las numéricas + `is_train` (data leakage). A pesar de esto, el modelo alcanzó AUC 0.9138 — con todas las features habilitadas probablemente superaría 0.95+.

### Causa del error del planner

El contrato base/scaffold del planner clasificó incorrectamente las features categóricas como forbidden. Esto es un error de razonamiento del LLM: confundió "features que necesitan encoding" con "features prohibidas".

---

## Causa Raíz #3: Bug menor del agente (solo intento 1)

**Error:** `log_loss(y_val, ensemble_pred, eps=1e-15)` — parámetro `eps` eliminado en sklearn moderno.

**Impacto:** Solo el intento 1. El agente corrigió correctamente en el intento 2 usando clipping manual. Este es un error menor de conocimiento de API, no de razonamiento.

---

## Veredictos de reviewers: Correctos dado el input incorrecto

| Reviewer | Veredicto | ¿Correcto? | Motivo real |
|----------|-----------|------------|-------------|
| **Reviewer** | REJECTED | Sí, dado `FAILED_RUNTIME` | No podía hacer otra cosa — el flag es determinístico |
| **QA Reviewer** | REJECTED (forzado) | Sí, pero notó que el código es correcto | Explícitamente dijo: "forced REJECTED due to deterministic blockers" |
| **Review Board** | REJECTED | Sí, dado los inputs | Cascada de los dos anteriores |

Los reviewers **no son la causa del problema**. Actuaron correctamente con la información que tenían.

---

## Resumen ejecutivo

| Categoría | Veredicto |
|-----------|-----------|
| **Razonamiento del ml_engineer** | **Correcto** — código funcional, AUC 0.9138 |
| **Reviewers** | **Correctos** — rechazaron por flags determinísticos |
| **Causa primaria** | **Bug en artifact validator** — no lee `scored_rows_schema.required_columns` |
| **Causa secundaria** | **Contrato del planner** — features categóricas prohibidas erróneamente |
| **Causa terciaria** | Bug menor de API (log_loss eps) — auto-corregido |
| **Fix más urgente** | Incluir `scored_rows_schema.required_columns` en allowed_cols del validator |
| **Fix secundario** | Revisar lógica del planner para no prohibir features categóricas |
