# Auditoría de Agentes Ingenieros — Data Engineer & ML Engineer

## Resumen Ejecutivo

Los agentes están **bien diseñados arquitectónicamente** pero tienen **debilidades operativas predecibles** que causan retries innecesarios. En la run exitosa (5b7afc9e), el DE necesitó 3 iteraciones y el ML falló 4 veces en sandbox — todos por errores de dtype que un ingeniero senior real no cometería. **No es limitación del LLM**; es falta de contexto concreto y guardas preventivas.

Nivel actual: **Mid-level engineer con buen framework** (~70% senior).
Con los cambios propuestos: **Senior engineer robusto** (~90%+ first-attempt success).

---

## Lo que YA está bien (nivel senior)

| Capacidad | Implementación | Archivo |
|---|---|---|
| Arquitectura contract-first | Precedencia clara: gates > contract > view > advisory | DE lines 362-365, ML lines 1846-1850 |
| Vistas proyectadas por agente | DE/ML views filtran solo lo relevante del contrato | `execution_planner.py` genera de_view/ml_view |
| Contexto de dependencias runtime | Allowlist de imports + versiones pinneadas | DE `_build_runtime_dependency_context()` |
| Compresión de contexto para wide-schema | `compress_long_lists()` + `COLUMN_LIST_POINTER` | `context_pack.py` |
| Soporte archivos en sandbox | `column_inventory.json`, `column_sets.json` disponibles | events.jsonl → support_files |
| Cleaning gates HARD/SOFT | Validación post-DE con cleaning_reviewer | `cleaning_reviewer.py` |
| Modo REPAIR estructurado | Feedback de errores anteriores + prioridad de fix | DE lines 206-221, ML lines 481-529 |
| Auto-fix pipeline | `json_dump_default_patch`, dialect autopatching | events.jsonl → auto_fix_applied |
| Protocolo senior | Decision log / Assumptions / Trade-offs / Risk register | `senior_protocol.py` |

---

## Debilidades Críticas y Propuestas de Mejora

### 1. NO HAY MUESTRA DE DATOS EN EL PROMPT (Impacto: ALTO)

**Problema**: El LLM nunca ve valores reales. Solo recibe metadata textual ("pixel columns are stored as objects", "47.4% null rate in label"). Esto causa que genere código con suposiciones incorrectas sobre dtypes, formatos y valores.

**Evidencia**: Run 5b7afc9e — DE iteration 1 falló porque asumió que label era siempre numérico y usó `.astype('int64')` directamente, cuando 28,000 rows del test set tienen label = NaN. Un senior que VE los datos hace `pd.to_numeric(errors='coerce').astype('Int64')` desde el primer intento.

**Propuesta**: Agregar al prompt del DE y ML un bloque concreto:
```python
# En data_engineer.py, dentro de generate_cleaning_script(), después de cargar el data_audit:
import pandas as pd
df_sample = pd.read_csv(input_path, dtype=str, nrows=5, sep=csv_sep, encoding=csv_encoding)
data_sample_block = f"""
DATA SAMPLE (first 5 rows, read as dtype=str):
{df_sample.to_string()}

ACTUAL DTYPES IN RAW FILE:
{df_sample.dtypes.to_string()}

SHAPE: {df_sample.shape}
NULL PATTERN (first 5 rows): 
{df_sample.isnull().sum().to_string()}
"""
```
Y renderizarlo en `$data_sample` dentro del SYSTEM_TEMPLATE.

**Coste**: ~500 tokens extra. **Beneficio**: Elimina ~80% de fallos de dtype en primera iteración.

---

### 2. PROMPT DE ML DEMASIADO LARGO: 110KB (Impacto: ALTO)

**Problema**: El prompt del ML engineer es 110,737 bytes. Con minimax-m2.5 (128K context), el prompt + respuesta (~20K) ya ocupa la mayoría del contexto. Peor: hay contexto duplicado.

**Evidencia**: `execution_contract_json` y `execution_contract_context` son idénticos (ambos llaman `json.dumps(execution_contract_compact, indent=2)`). Las líneas 2060-2061 del ml_engineer.py:
```python
execution_contract_json=json.dumps(execution_contract_compact, indent=2),
execution_contract_context=json.dumps(execution_contract_compact, indent=2),
```

**Propuesta**:
- Eliminar uno de los dos contract contexts duplicados
- Comprimir `column_summaries` más agresivamente (120 items × ~200 bytes cada uno = 24KB solo en summaries)
- Para columnas homogéneas (los 720 pixel columns tienen null_frac=0, dtype=int64), usar un resumen agregado en lugar de listar individualmente
- Objetivo: reducir prompt a <50KB

---

### 3. NO HAY LISTA PRE-EXPANDIDA DE FEATURES (Impacto: MEDIO-ALTO)

**Problema**: El prompt dice "Expand required_feature_selectors against input header" pero el LLM debe implementar la lógica de regex expansion desde cero. En datasets anchos, esto falla si el LLM no sabe exactamente qué columnas matchean.

**Evidencia**: El DE prompt muestra `"required_feature_selectors": [{"type": "regex", "pattern": "^pixel\\d+$"}]` pero nunca dice explícitamente "esto matchea 784 columnas: pixel0, pixel1, ..., pixel783".

**Propuesta**: En el orquestador, pre-expandir los selectors y pasar un resumen:
```json
{
  "expanded_selectors_summary": {
    "PIXEL_FEATURES": {
      "selector": {"type": "regex", "pattern": "^pixel\\d+$"},
      "matched_count": 784,
      "sample": ["pixel0", "pixel1", "pixel2", "...", "pixel783"],
      "expansion_code": "pixel_cols = [c for c in df.columns if re.match(r'^pixel\\d+$', c)]"
    }
  }
}
```

---

### 4. NO HAY CONTRATO DE DTYPES POR COLUMNA (Impacto: MEDIO-ALTO)

**Problema**: El contrato dice qué columnas son requeridas y sus roles, pero nunca dice qué dtype deben tener POST-limpieza. El LLM adivina.

**Evidencia**: DE iteration 1 usó `int64` para label (falla con NaNs), iteration 3 usó `Int64` (correcto). ML usó `np.bincount(y_train)` sin `.astype(int)` porque no sabía que label era float64 después de la limpieza.

**Propuesta**: Agregar al contrato un `column_dtype_contract`:
```json
{
  "column_dtype_targets": {
    "label": {"target_dtype": "Int64", "note": "Nullable integer, 40% nulls in test set"},
    "pixel*": {"target_dtype": "int64", "note": "0-255 range, no nulls expected"},
    "__split": {"target_dtype": "object", "note": "Categorical: train/test"}
  }
}
```

---

### 5. FEEDBACK DE REPAIR TRUNCADO (Impacto: MEDIO)

**Problema**: El traceback de error se corta a los ~200 caracteres, perdiendo la línea exacta del error.

**Evidencia**: DE iteration 2 context:
```
File "/tmp/run/ml_script.py", line 133, in numeric_convers  ← CORTADO
```
El nombre de la función está truncado, y el mensaje de error real no aparece.

**Propuesta**: En el feedback_context, incluir:
- Traceback completo (últimas 20 líneas)
- Las líneas de código ±10 alrededor del error
- Tipo de excepción + mensaje completo

---

### 6. NO HAY PREFLIGHT LOCAL (Impacto: MEDIO)

**Problema**: El código va directo a Cloud Run (2-3 min por intento). Si tiene SyntaxError o ImportError, se pierden minutos.

**Evidencia**: Ya existe `ast.parse()` en `_clean_code()`, pero no hay check de imports contra el allowlist. El ML script importa `joblib` — si no estuviera en el sandbox, serían 3 minutos perdidos.

**Propuesta**: Agregar un preflight check:
```python
def _preflight_check(code: str, allowlist: set) -> List[str]:
    tree = ast.parse(code)
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.')[0]
                if root not in allowlist:
                    issues.append(f"Import '{root}' not in sandbox allowlist")
    return issues
```

---

### 7. PROTOCOLO SENIOR DEMASIADO GENÉRICO (Impacto: MEDIO)

**Problema**: `SENIOR_ENGINEERING_PROTOCOL` son 8 líneas genéricas. Un senior real tiene patterns específicos para datos tabulares.

**Propuesta**: Agregar patterns concretos al protocolo:
```
DTYPE SAFETY PATTERNS:
- Read CSV with dtype=str, then convert with pd.to_numeric(errors='coerce')
- For integer columns with possible nulls, use nullable Int64 dtype
- Never use .astype(int/float) without handling NaN first
- Always verify selector/regex matched >0 columns before proceeding

OUTPUT SAFETY PATTERNS:
- Verify all required output directories exist before writing
- After writing CSV, verify row count matches expected
- JSON serialization: always handle numpy/pandas types with custom default
- Never assume column order; use explicit column selection

WIDE DATASET PATTERNS:
- Load column_sets.json at runtime to resolve family selectors
- Prefer column-selection by pattern (regex) over explicit listing
- For 100+ homogeneous features, apply transformations vectorized, not column-by-column
```

---

### 8. NO HAY SKELETON/TEMPLATE DE SCRIPT (Impacto: MEDIO)

**Problema**: El LLM genera el script entero desde cero cada vez, incluyendo boilerplate que siempre es igual (imports, directory creation, JSON serialization, CSV dialect loading). Esto introduce variabilidad innecesaria.

**Propuesta**: Pre-inyectar un skeleton que el LLM solo necesita completar:
```python
# Ya existe parcialmente: la variable `injection` en DE (lines 521-589)
# Pero solo cubre json/numpy/pandas safety. Extender con:
# - Lectura de cleaning_manifest.json para dialect
# - Creación de directorios de output
# - Helper para expandir selectors desde column_sets.json
# - Template de logging/decision-log
```

---

### 9. COLUMN SUMMARIES TRUNCADOS PARA ML (Impacto: BAJO-MEDIO)

**Problema**: ML recibe solo 120 de 721 column summaries (17%). Las 601 omitidas son pixel columns homogéneas, pero el LLM no sabe que son homogéneas.

**Propuesta**: Para columnas agrupadas por selector family, agregar un resumen agregado:
```json
{
  "family_aggregate": {
    "PIXEL_FEATURES": {
      "count": 720,
      "dtype_observed": "int64 (all)",
      "null_frac": "0.0 (all)",
      "value_range": "[0, 255]",
      "constant_columns_dropped": 65
    }
  }
}
```

---

## Resumen de Prioridades

| # | Mejora | Impacto | Esfuerzo | ROI |
|---|---|---|---|---|
| 1 | Data sample en prompt | ALTO | Bajo (20 LOC) | ★★★★★ |
| 2 | Reducir prompt ML a <50KB | ALTO | Medio (refactor) | ★★★★☆ |
| 3 | Pre-expandir selectors | MEDIO-ALTO | Bajo (30 LOC) | ★★★★☆ |
| 4 | Column dtype contract | MEDIO-ALTO | Medio (planner change) | ★★★★☆ |
| 5 | Feedback de repair completo | MEDIO | Bajo (15 LOC) | ★★★☆☆ |
| 6 | Preflight local | MEDIO | Bajo (20 LOC) | ★★★☆☆ |
| 7 | Protocolo senior específico | MEDIO | Bajo (texto) | ★★★☆☆ |
| 8 | Script skeleton/template | MEDIO | Medio (diseño) | ★★★☆☆ |
| 9 | Family aggregate summaries | BAJO-MEDIO | Bajo (20 LOC) | ★★☆☆☆ |

**Quick wins (1 hora de trabajo)**: Items 1, 3, 5, 6, 7 — reducirían retries en ~60%.
**Medium effort (medio día)**: Items 2, 4 — mejorarían calidad de primera iteración significativamente.
**Longer term (1 día)**: Item 8 — cambio arquitectónico pero mayor robustez a largo plazo.