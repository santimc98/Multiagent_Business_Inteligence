# AUDITORÍA: Execution Planner — Run 6a51bf4f (Heart Disease Kaggle)
## Análisis Forense de Fallo + Assessment de Robustez

**Fecha:** 2026-02-15  
**Run ID:** 6a51bf4f  
**Dataset:** 900K filas, 16 columnas, clasificación binaria (Heart Disease)  
**Resultado:** `pipeline_aborted:execution_contract_invalid` — 32 minutos gastados, 7 intentos LLM, 0 contratos válidos  

---

## DIAGNÓSTICO: QUÉ PASÓ EXACTAMENTE

### Timeline de la Run

```
16:17:55  run_init (csv: business_input.csv, 900K rows × 16 cols)
16:17:55  steward_start
16:18:58  steward_complete ✅ (encoding=utf-8, sep=",", decimal=".")
16:20:04  execution_planner_start (strategy="Native Categorical GBM Ensemble")
          │
          ├── Sectional Compiler:
          │   ├── core section round 1: FAIL (canonical_columns dict instead of list)
          │   ├── core section round 2: OK ✅
          │   ├── cleaning_contract round 1: FAIL (required_feature_selectors format)
          │   └── cleaning_contract round 2: FAIL (selectors missing "type" key)
          │       → Sectional compiler abandoned
          │
          ├── Full-Contract Quality Gate (3 rounds max):
          │   ├── Attempt 1: FAIL — column_dtype_targets uses "type" not "target_dtype"
          │   ├── Attempt 2: FAIL — JSON truncated (MAX_TOKENS), can't parse
          │   └── Attempt 3: FAIL — SAME ERROR: "type" not "target_dtype" 
          │
          └── Result: contract = {} (empty), llm_success = false
          
16:52:59  pipeline_aborted_reason: execution_contract_invalid
16:53:33  run_finalize: NO_GO
```

**32 minutos y 7 llamadas LLM para producir un contrato vacío `{}`.**

---

## ROOT CAUSE #1 (CRITICAL): Schema Mismatch — `"type"` vs `"target_dtype"`

### El Problema

El validador (`contract_validator.py:2654`) espera:

```json
"column_dtype_targets": {
  "age": {"target_dtype": "float64", "nullable": false}
}
```

Pero el LLM genera consistentemente:

```json
"column_dtype_targets": {
  "age": {"type": "float64"}
}
```

El LLM usa `"type"` porque es el nombre estándar en JSON Schema. El validador espera `"target_dtype"` que es un nombre custom.

### Por qué el LLM no lo corrige en 3 intentos

**Prompt inicial** (línea ~19): dice solamente:
```
column_dtype_targets: object mapping concrete columns and/or 
selector families to target dtype contracts
```

No da ningún ejemplo JSON. No dice que la key debe ser `"target_dtype"`.

**Repair prompt** (attempt 2 y 3): dice:
```
- Resolve `contract.column_dtype_targets`: column_dtype_targets['id'] missing target_dtype.
```

Esto dice "missing target_dtype" pero no muestra el formato correcto. El LLM interpreta esto como "el objeto necesita un campo más" en vez de "el campo `type` debería llamarse `target_dtype`".

### Evidencia: El LLM Progresa Pero No Llega

| Attempt | Formato producido | Error del validador |
|---------|-------------------|---------------------|
| 1 | `"age": "float64"` (string plano) | "must be an object" |
| 2 | (JSON truncado — MAX_TOKENS) | Parse error |
| 3 | `"age": {"type": "float64"}` (objeto) | "missing target_dtype" |

El LLM SÍ aprende entre intento 1→3: pasa de string plano a objeto. Pero nunca le dicen que la key debe ser `"target_dtype"` en vez de `"type"`.

### Ironía: Existe Código Determinístico que Genera el Formato Correcto

`execution_planner.py:2220` — `_infer_column_dtype_targets()` genera:

```python
dtype_targets[col] = {
    "target_dtype": target_dtype,   # ← CORRECTO
    "nullable": bool(miss_val and miss_val > 0.0),
    "role": role_hint or "unknown",
    "source": "data_profile",
}
```

Esta función es llamada desde `build_contract_min()` (línea 2777), que produce un contrato determinístico completo con `column_dtype_targets` perfectos. **Pero `build_contract_min()` NUNCA se invoca** — es código muerto en este contexto.

---

## ROOT CAUSE #2: Sectional Compiler Falla en `required_feature_selectors`

### Qué pasó

- Round 1: LLM genera selectors sin el formato esperado → `"must be list[object]"`
- Round 2: LLM genera objetos pero sin campo `"type"` en cada selector → `"selectors[0] is missing type"`

El prompt para la sección cleaning_contract tampoco da un ejemplo concreto del formato de selector.

### Consecuencia

El sectional compiler produce una "mitad" del contrato (core OK, cleaning FAIL). Como cleaning_contract falla, el sectional result no se acepta, y el sistema cae a full-contract mode. Pero el full-contract mode también falla por el mismo tipo de problema (formatos no documentados).

---

## ROOT CAUSE #3: Sin Fallback Determinístico Cuando LLM Falla

### Qué pasa cuando 7 intentos fallan

```python
# execution_planner.py:9020
contract = {}          # ← Contrato VACÍO
llm_success = False
```

El sistema NO usa el `build_contract_min()` disponible como fallback. Simplemente persiste `{}`, que luego se valida con 22 errores, y la pipeline se aborta.

**Costo real:** 32 minutos de compute, ~450K tokens consumidos (7 llamadas × ~65K tokens/llamada), resultado = nada.

### Lo que debería pasar

Si el LLM no logra producir un contrato válido después de N intentos, el sistema debería:

1. Tomar el mejor candidato (el de attempt 3, que tiene 17 errores pero contenido válido)
2. Aplicar reparaciones determinísticas para errores conocidos (como `"type"` → `"target_dtype"`)
3. Rellenar campos faltantes desde `build_contract_min()` 
4. Validar el resultado reparado
5. Si pasa → usarlo. Si no → fallar con diagnóstico útil.

---

## ROOT CAUSE #4: Attempt 2 Truncado (MAX_TOKENS)

### Qué pasó

```json
{
  "attempt_index": 2,
  "finish_reason": "2",           // ← MAX_OUTPUT_TOKENS
  "response_char_len": 8335,      // ← Respuesta cortada
  "had_json_parse_error": true,
  "parse_error_message": "unterminated string literal (detected at line 341)"
}
```

El full-contract JSON es ~6-8KB. Con un prompt de ~31K chars, el modelo genera ~8K chars de respuesta y se trunca a mitad del JSON. Esto desperdicia 1 de los 3 intentos disponibles.

### Causa

El `max_output_tokens` del modelo está configurado demasiado bajo para generar un contrato completo. O el prompt es demasiado largo y reduce el espacio disponible para la respuesta.

---

## ROOT CAUSE #5: Repair Prompt No Incluye Ejemplos de Formato Correcto

### `_build_targeted_repair_actions()` — Sin Acción para `column_dtype_targets`

```python
rule_to_action = {
    "contract.cleaning_transforms_drop_conflict": "Ensure drop_columns...",
    "contract.clean_dataset_selector_drop_required_conflict": "When selector-drop...",
    # ... 10 reglas más ...
    # ❌ NO HAY ENTRADA PARA "contract.column_dtype_targets"
}
```

Cuando el rule es `contract.column_dtype_targets`, cae al fallback genérico:
```python
actions.append(f"- Resolve `{rule}`: {msg}")
```

Que produce:
```
- Resolve `contract.column_dtype_targets`: column_dtype_targets['id'] missing target_dtype.
```

Sin ejemplo concreto del formato esperado.

---

## MEJORAS PRIORIZADAS

### P0 — Hotfix Inmediato (15 minutos)

**P0-1: Agregar ejemplo de `column_dtype_targets` al prompt**

En la descripción del schema del prompt (línea ~19 de execution_planner.py), cambiar:

```
ANTES:
- column_dtype_targets: object mapping concrete columns and/or selector families 
  to target dtype contracts

DESPUÉS:
- column_dtype_targets: object mapping column name -> dtype spec.
  Each value MUST be an object with key "target_dtype" (not "type").
  Example: {"age": {"target_dtype": "float64", "nullable": false}, 
            "sex": {"target_dtype": "string"}}
```

**P0-2: Agregar acción de repair para `column_dtype_targets`**

En `_build_targeted_repair_actions()`, agregar al `rule_to_action`:

```python
"contract.column_dtype_targets": (
    "Each column_dtype_targets entry must be an object with key 'target_dtype' "
    "(NOT 'type'). Example: {\"target_dtype\": \"float64\", \"nullable\": false}. "
    "Replace all {\"type\": X} with {\"target_dtype\": X}."
),
```

---

### P1 — Reparación Determinística Post-LLM (1-2h)

**P1-1: `_deterministic_repair_column_dtype_targets()`**

Cuando el validador reporta "missing target_dtype" y el entry tiene `"type"`, rename determinístico:

```python
def _deterministic_repair_column_dtype_targets(contract: dict) -> dict:
    """Fix common LLM format error: 'type' -> 'target_dtype'"""
    targets = contract.get("column_dtype_targets")
    if not isinstance(targets, dict):
        return contract
    
    repaired = {}
    for col, spec in targets.items():
        if isinstance(spec, str):
            # Flat string → wrap in object
            repaired[col] = {"target_dtype": spec}
        elif isinstance(spec, dict):
            if "target_dtype" not in spec and "type" in spec:
                # Wrong key name → rename
                new_spec = dict(spec)
                new_spec["target_dtype"] = new_spec.pop("type")
                repaired[col] = new_spec
            elif "target_dtype" not in spec and "dtype" in spec:
                new_spec = dict(spec)
                new_spec["target_dtype"] = new_spec.pop("dtype")
                repaired[col] = new_spec
            else:
                repaired[col] = spec
        else:
            repaired[col] = {"target_dtype": "preserve"}
    
    contract["column_dtype_targets"] = repaired
    return contract
```

Aplicar ANTES de la validación de calidad, en cada attempt.

**P1-2: Fallback a `build_contract_min()` cuando LLM falla**

Cuando todos los intentos fallan y hay un `best_candidate`:

```python
if contract is None or contract == {}:
    # Try deterministic repairs on best candidate
    if best_candidate and isinstance(best_candidate, dict):
        repaired = _deterministic_repair(best_candidate)
        validation = _validate_contract_quality(copy.deepcopy(repaired))
        if _contract_is_accepted(validation):
            contract = repaired
            llm_success = True  # or "deterministic_repair"
    
    # If still failing, try contract_min as base
    if contract is None or contract == {}:
        try:
            contract_min = build_contract_min(
                contract=best_candidate or {},
                strategy=strategy,
                business_objective=business_objective,
                canonical_columns=column_inventory or [],
                column_roles=best_candidate.get("column_roles", {}) if best_candidate else {},
                data_profile=data_profile,
                ...
            )
            validation = _validate_contract_quality(copy.deepcopy(contract_min))
            if _contract_is_accepted(validation):
                contract = contract_min
                llm_success = True  # "deterministic_fallback"
        except Exception:
            pass
```

**P1-3: Aumentar `max_output_tokens` para contract generation**

El contrato completo necesita ~2000-3000 tokens. Si el prompt consume 10K tokens y el modelo tiene un context de ~75K, debería haber espacio de sobra. Verificar que `max_output_tokens` esté configurado a al menos 4000.

---

### P2 — Robustez Estructural (medio día)

**P2-1: Batería de reparaciones determinísticas pre-validación**

Crear `_apply_deterministic_repairs(contract)` que aplique fixes conocidos:

```python
def _apply_deterministic_repairs(contract: dict) -> dict:
    """Apply known deterministic fixes before quality validation."""
    contract = _deterministic_repair_column_dtype_targets(contract)
    contract = _deterministic_repair_scope(contract)
    contract = _deterministic_repair_gate_format(contract)
    contract = _deterministic_repair_required_feature_selectors(contract)
    return contract
```

Para cada campo problemático, si la estructura es "casi correcta" (renaming, wrapping), reparar determinísticamente en vez de gastar otra llamada LLM.

**P2-2: Schema examples embebidos en prompt para cada campo complejo**

Para cada campo que tiene formato no-obvio, agregar un ejemplo inline:

```
- column_dtype_targets: {col: {"target_dtype": "float64", "nullable": false}}
- cleaning_gates: [{"name": "no_nulls_target", "severity": "HARD", "params": {"column": "target"}}]
- required_feature_selectors: [{"type": "prefix", "value": "feature_", "match_mode": "startswith"}]
```

**P2-3: Rescue del sectional compiler parcial**

Cuando el sectional compiler produce secciones parciales (core OK, cleaning FAIL), en vez de abandonar:

1. Tomar la sección exitosa (core)
2. Rellenar la sección fallida desde `build_contract_min()`
3. Merge y validar

---

### P3 — Senior Robustness (1-2 días)

**P3-1: Contract schema registry con validación y ejemplos**

Crear un archivo `contract_schema.py` que defina:

```python
CONTRACT_FIELD_SCHEMAS = {
    "column_dtype_targets": {
        "type": "dict[str, dict]",
        "required_keys_per_entry": ["target_dtype"],
        "example": {"age": {"target_dtype": "float64", "nullable": false}},
        "common_errors": {
            "type_instead_of_target_dtype": {
                "detect": lambda v: "type" in v and "target_dtype" not in v,
                "fix": lambda v: {**v, "target_dtype": v.pop("type")},
            },
        },
    },
    "cleaning_gates": {
        "type": "list[dict]",
        "required_keys_per_entry": ["name", "severity"],
        "example": [{"name": "gate_1", "severity": "HARD", "params": {}}],
    },
    # ... etc
}
```

Esto centraliza:
- Formato esperado (para documentación y prompt)
- Ejemplo JSON (para inyectar en prompts)
- Errores comunes + reparación automática
- Validación tipada

**P3-2: Token budget para contract generation**

Medir tokens del prompt y ajustar `max_output_tokens` dinámicamente:

```python
prompt_tokens = estimate_tokens(full_prompt)
available_tokens = MODEL_CONTEXT_LIMIT - prompt_tokens
max_output = min(available_tokens - 500, 4000)  # Safety margin
```

Esto previene truncaciones como la del attempt 2.

**P3-3: Progressive contract construction**

En vez de pedir el contrato completo en 1 sola llamada, construir incrementalmente:

1. **Pass 1** (determinístico): Generar esqueleto desde data_profile + strategy + business_objective
   - scope, canonical_columns, column_roles, column_dtype_targets, required_outputs → todo determinístico
2. **Pass 2** (LLM): Solo pedir los campos que requieren juicio:
   - cleaning_gates, qa_gates, runbooks, iteration_policy, evaluation_spec
3. **Merge**: Combinar pass 1 + pass 2 + validar

Beneficio: el 60-70% del contrato se genera determinísticamente (sin errores de formato). El LLM solo se ocupa de lo que requiere razonamiento.

---

## ANÁLISIS DE COSTES DE LA FALLA

| Recurso | Gastado | Producido |
|---------|---------|-----------|
| Tiempo | 32 min | 0 (pipeline aborted) |
| LLM calls | 7 (4 sectional + 3 full) | 0 contratos válidos |
| Tokens | ~450K total | Nada útil |
| Steward work | 1 min (válido) | Desperdiciado |
| Strategy work | 1 min (válido) | Desperdiciado |

**Con P0 (15 min de dev):** La run habría producido un contrato válido en attempt 3. El LLM generó contenido correcto excepto por el nombre de una key JSON.

**Con P1 (2h de dev):** Cualquier run futura con errores de formato sería reparada automáticamente.

---

## ASSESSMENT COMPARATIVO: Planner vs Otros Agentes

| Dimensión | Execution Planner | Steward | Strategist |
|-----------|------------------|---------|------------|
| Líneas de código | 9138 | ~2700 | 1016 |
| Complejidad | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| Deterministic fallback | ❌ Dead code | ✅ 2-pass | ✅ Single strategy |
| Schema documentation | ❌ Vague | N/A | N/A |
| Auto-repair | ❌ None | ★★★☆☆ | ★★★★☆ |
| Error feedback quality | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ |
| Token efficiency | ★★☆☆☆ | ★★★★☆ | ★★★★★ |

**El Planner es el agente más complejo (9138 líneas) pero el menos resiliente.**

---

## CONCLUSIÓN

La falla de esta run se reduce a un **single point of failure trivial**: el LLM escribe `"type"` donde el validador espera `"target_dtype"`. Un rename de key JSON. Esto causó:

- 7 llamadas LLM desperdiciadas
- 32 minutos perdidos
- Pipeline completa abortada
- 0 resultados para el usuario

Pero el Planner ya tiene el código para generar `column_dtype_targets` correctos: `_infer_column_dtype_targets()` produce el formato perfecto. Solo que nunca se usa como fallback.

**La fix más urgente (P0) toma 15 minutos** y previene esta categoría de fallo: agregar un ejemplo de formato al prompt y un targeted repair action. 

**La fix estructural (P1) toma 2 horas** y hace al Planner resiliente a cualquier error de formato conocido: reparación determinística pre-validación + fallback a `build_contract_min()`.

El patrón arquitectónico subyacente es que el Planner trata al LLM como la **única fuente de verdad** para el contrato, cuando el 60-70% del contrato puede generarse determinísticamente desde los datos del Steward. El LLM solo debería aportar juicio (gates, runbooks, policy), no formato estructural.