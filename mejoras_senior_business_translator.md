# AUDITORÍA: Business Translator Agent
## Agente de Reporte Ejecutivo — Nivel Senior Assessment

**Fecha:** 2026-02-15  
**Scope:** `business_translator.py` (1915L), `pdf_generator.py` (187L), `senior_protocol.py` (73L), `contract_views.py` (translator_view), integración `graph.py`  
**Rol esperado:** Senior Executive Translator — genera el reporte final para CEO/directivos traduciendo resultados técnicos a narrativa de negocio.

---

## VEREDICTO GLOBAL

| Dimensión | Score | Nota |
|-----------|-------|------|
| Context Assembly (recopilar evidencia) | 82% | Excelente — 30+ fuentes, fact cards, manifest, tablas |
| Prompt Engineering | 58% | Ambicioso pero sobrecargado, ~230 líneas de instrucciones |
| Output Validation | 12% | **Crítico** — casi zero verificación post-generación |
| Error Recovery | 25% | Fallback es un string plano, sin reporte determinístico |
| PDF Pipeline | 45% | Funcional pero desconecta imágenes del contexto narrativo |
| **GLOBAL** | **62% Mid-Senior** | Fuerte en preparación, débil en verificación de resultados |

**Analogía:** Es como un consultor senior que prepara un dossier impecable de 50 páginas de evidencia... pero entrega el informe final sin releerlo.

---

## ARQUITECTURA

```
graph.py::run_translator()
    │
    ├── Promueve best_attempt si último intento falló
    ├── Construye run_summary, output_contract_report
    ├── Resuelve translator_view (reporting_policy + evidence_inventory)
    ├── Genera recommendations_preview si aplica
    │
    └── translator.generate_report(state, error_message, plots, translator_view)
            │
            ├── [1] CONTEXT ASSEMBLY (~800 líneas)
            │     ├── Carga 12+ JSON artifacts (metrics, insights, run_summary...)
            │     ├── Carga CSVs (cleaned_dataset, predictions)
            │     ├── 15+ funciones _summarize_*() compactan contexto
            │     ├── _build_fact_cards() → hechos determinísticos
            │     ├── _build_report_artifact_manifest() → inventario + compliance
            │     ├── Tablas HTML (inventory, compliance, KPI snapshot)
            │     ├── Tablas ASCII (metrics, cleaned_sample, scored_sample)
            │     ├── _derive_exec_decision() → GO/NO_GO/GO_WITH_LIMITATIONS
            │     └── Slot coverage analysis
            │
            ├── [2] PROMPT CONSTRUCTION (~230 líneas template)
            │     ├── System prompt con 35+ variables sustituidas
            │     ├── SENIOR_TRANSLATION_PROTOCOL + EVIDENCE_RULE
            │     ├── Formatting constraints, narrative style guide
            │     ├── Section-by-section content requirements
            │     └── User message con execution_results
            │
            ├── [3] LLM CALL (1 sola llamada)
            │     └── gemini-3-flash-preview, temperature=0.2
            │
            └── [4] POST-PROCESSING (mínimo)
                  ├── _sanitize_report_text() → limpia "..."
                  ├── _ensure_evidence_section() → fuerza "## Evidencia usada"
                  ├── sanitize_text() → encoding
                  └── return content (sin validación)
```

---

## FORTALEZAS DETALLADAS

### ✅ S1: Context Assembly masivo y bien organizado (líneas 1069-1587)

La preparación de contexto es lo mejor del agente. Carga y compacta 30+ fuentes de evidencia:

- **12+ JSON artifacts** cargados directamente del filesystem (metrics.json, insights.json, run_summary.json, data_adequacy_report.json, alignment_check.json, etc.)
- **CSV artifacts** procesados con dialect-aware loading (cleaned_dataset, predictions)
- **15+ funciones `_summarize_*()`** que compactan cada fuente a lo esencial
- **Artifact manifest** con compliance checks (required vs present, status badges)
- **Tablas HTML pre-renderizadas** para inventario, compliance, y KPI snapshot
- **Tablas ASCII** para datos tabulares (compatible PDF)
- **Slot coverage analysis** que detecta required slots sin payload

Esto garantiza que el LLM tiene acceso a toda la evidencia disponible.

### ✅ S2: Executive Decision derivada determinísticamente (líneas 868-889)

`_derive_exec_decision()` calcula GO/NO_GO/GO_WITH_LIMITATIONS usando lógica determinística basada en:
- review_verdict (NEEDS_IMPROVEMENT, REJECTED → NO_GO)
- data_adequacy status (insufficient_signal → GO_WITH_LIMITATIONS)  
- lift value (si ≤ 0 con data_limited → NO_GO)
- sufficient_signal + lift > 0 → GO

El LLM recibe esta decisión como `executive_decision_label` y debe usarla — no inventarla.

### ✅ S3: Fact Cards determinísticos (líneas 1003-1051)

`_build_fact_cards()` extrae métricas verificables de:
- case_summary (top examples por valor)
- scored_rows (predictions con mayor score)
- weights_payload (métricas de modelos)
- data_adequacy (señales de calidad)

Cada fact tiene `{source, metric, value, labels}` — trazable a su artifact de origen.

### ✅ S4: Smart Column Selection (líneas 676-741)

`_select_informative_columns()` prioriza columnas para las tablas sample:
- Separa numeric (ratio ≥ 0.6) vs categorical
- Ordena numeric por completitud, categorical por cardinalidad
- Garantiza mínimo 5 columnas en muestra

`_select_scored_columns()` busca columnas de predicción por tokens (pred, prob, score, segment, cluster, rank, risk).

### ✅ S5: Evidence Section enforcement (líneas 959-990)

`_ensure_evidence_section()` garantiza que el reporte siempre termina con "## Evidencia usada" con:
- Items `{claim, source}` mínimo 3
- Lista de artifact paths verificados
- Normalización de sources (invalidas → "missing")
- Reemplaza sección LLM-generada si existe (para consistencia)

### ✅ S6: Language Detection automática (líneas 26-71)

Detecta español vs inglés por stopword overlap en el business_objective. El reporte completo se genera en el idioma detectado.

### ✅ S7: Prompt con principios de consultoría senior (líneas 1615-1629)

El "SENIOR CONSULTANT NARRATIVE STYLE" es excelente:
- No slot-filling (adaptar, no rellenar)
- No placeholders ("Not available")
- Flujo coherente entre secciones
- Brevedad ejecutiva
- Tono decidido

### ✅ S8: Error path definido (líneas 1716-1733)

Si hay error, el prompt especifica formato de "EXECUTION FAILURE REPORT" con:
- Status line con emoji ⛔
- Explicación no-técnica
- Artifacts faltantes
- Visual evidence si existe
- Next actions para desbloquear

---

## DEBILIDADES CRÍTICAS

### ❌ BT-1 (CRITICAL): ZERO validación post-generación del reporte

**Evidencia:** Líneas 1905-1914:
```python
response = self.model.generate_content(full_prompt)
content = (getattr(response, "text", "") or "").strip()
content = _sanitize_report_text(content)      # Solo limpia "..."
content = _ensure_evidence_section(content, evidence_paths)  # Solo sección evidencia
content = sanitize_text(content)              # Solo encoding
self.last_response = content
return content                                # ← SIN VALIDACIÓN
```

**No se verifica:**
- ¿Existen todas las secciones requeridas (Executive Decision, Evidence, Risks)?
- ¿El executive_decision_label en el reporte coincide con el determinístico?
- ¿Las métricas citadas por el LLM coinciden con los valores reales de artifacts?
- ¿Los nombres de segmentos/categorías mencionados existen en los datos?
- ¿Los plots referenciados existen en plots_list?
- ¿El reporte tiene longitud razonable (no 2 líneas ni 20 páginas)?
- ¿Hay claims sin source en la sección de evidencia?

**Impacto:** Un CEO puede recibir un reporte con métricas inventadas, segmentos que no existen, o decisión contradictoria con los datos. Esto es inaceptable para un agente de nivel senior.

**Contraste:** El Steward tiene 2-pass con validación intermedia. El Strategist tiene repair loop de hasta 3 intentos. El Reviewer tiene checks determinísticos. El Translator: ZERO.

**Fix — `_validate_report(content, facts_context, executive_decision_label, metrics_payload, plots)` (~120 líneas):**

```
1. STRUCTURAL CHECKS:
   - Regex: ¿Existe "## Executive Decision" o "## Decisión Ejecutiva"?
   - Regex: ¿Existe "## Evidence" o "## Evidencia"?
   - Regex: ¿Existe "## Risks" o "## Riesgos"?
   - Longitud: 500 < len(content) < 30000 chars
   - Si falla → flag "incomplete_report"

2. DECISION CONSISTENCY:
   - Extraer primera mención de GO/NO_GO/GO_WITH_LIMITATIONS del reporte
   - Comparar con executive_decision_label
   - Si difiere → flag "decision_mismatch"

3. METRIC CROSS-VERIFICATION:
   - Extraer patrones numéricos del reporte: r"(\d+\.?\d*)\s*%" , r"[Rr]²?\s*[=:]\s*(\d+\.?\d*)"
   - Para cada métrica extraída, buscar valor similar (±5%) en facts_context/metrics_payload
   - Si no se encuentra → flag "unverified_metric: <value>"

4. PLOT REFERENCE CHECK:
   - Extraer ![...](path) del reporte
   - Verificar path está en plots_list
   - Si no → flag "invalid_plot_ref"

5. RESULT:
   - Si flags críticos (decision_mismatch, >2 unverified_metrics) → retry con prompt corregido
   - Si flags menores → append warning note al final del reporte
   - Si clean → return as-is
```

---

### ❌ BT-2 (HIGH): Prompt sobrecargado — 230 líneas de instrucciones + 35+ contextos

**Evidencia:** System prompt template: líneas 1590-1816 = **226 líneas de instrucciones**. Más 35+ bloques de contexto inyectados (líneas 1837-1876).

**Problemas de instruction fatigue:**

1. **Instrucciones contradictorias:**
   - Línea 1815: `OUTPUT: Markdown format (NO TABLES).`
   - Línea 1682: `Prefer HTML tables for executive readability`
   - Línea 1607: `NO MARKDOWN TABLES` (pero sí HTML tables)
   - El LLM debe navegar 3 reglas distintas sobre tablas.

2. **Contextos redundantes:**
   - `data_adequacy_context` (summarized) + `data_adequacy_report_json` (verbatim) = misma info 2x
   - `weights_context` + `model_metrics_context` = solapan significativamente
   - `artifacts_context` + `artifact_manifest_context` + `artifact_inventory_table_html` = 3 vistas del mismo inventario
   - `insights_context` incluye métricas que ya están en `facts_context`

3. **Contextos raw JSON sin procesar:**
   - Líneas 1837-1876: `json.dumps(gate_context)`, `json.dumps(steward_context)`, etc.
   - El LLM debe parsear JSON crudo para extraer información
   - Un resumen textual sería mucho más eficiente en tokens

**Estimación de tokens del prompt:** 15K-50K tokens dependiendo del run. Un prompt más largo ≠ mejor reporte. La investigación muestra que después de ~8K tokens de instrucciones, los LLMs empiezan a perder adherencia a instrucciones posteriores.

**Fix — Consolidar y estructurar en 2 niveles:**

Nivel 1 — **FACTS_BLOCK** (determinístico, ~2K tokens): Los hechos duros que el LLM NO debe modificar.
```
EXECUTIVE_DECISION: GO_WITH_LIMITATIONS
METRICS: {accuracy: 0.847, f1: 0.82, lift: +12.3%}
DATA_ADEQUACY: data_limited (reasons: small_sample, no_baseline)
REVIEW: APPROVED_WITH_WARNINGS
ARTIFACTS: 8/10 required present
TOP_CORRELATIONS: [price→target: 0.72, quantity→target: 0.65]
```

Nivel 2 — **NARRATIVE_GUIDE** (~80 líneas): Instrucciones de estilo y estructura.

Nivel 3 — **CONTEXT_APPENDIX**: Detalles de soporte (solo si se necesitan).

---

### ❌ BT-3 (HIGH): Una sola llamada LLM — sin recovery ni retry

**Evidencia:** Línea 1906: `response = self.model.generate_content(full_prompt)` — 1 llamada única.

**Contraste con otros agentes:**
- Steward: 2-pass (hypothesis → evidence → finalize)
- Strategist: repair loop de hasta 3 intentos con scoring
- Domain Expert (antes de eliminarlo): al menos tenía JSON forzado

**Escenarios sin recovery:**
- LLM genera reporte de 3 líneas → se entrega así
- LLM alucina métricas → no se detecta
- LLM omite "Risks & Limitations" → no se detecta
- LLM mezcla idiomas → no se detecta
- API error parcial (respuesta truncada) → solo se captura si Exception

**Fix — 2 opciones:**

**Opción A: Validation + Targeted Repair (recomendado)**
```python
report = self.model.generate_content(full_prompt)
issues = _validate_report(report, facts_context, executive_decision_label, ...)
if issues.critical:
    repair_prompt = _build_repair_prompt(report, issues)
    report = self.model.generate_content(repair_prompt)  # Retry 1x
    issues = _validate_report(report, ...)
    if issues.critical:
        report = _generate_deterministic_fallback(facts_context, executive_decision_label, ...)
```

**Opción B: 2-Pass Generation**
- Pass 1: Generar outline estructurado con hechos clave (JSON)
- Pass 2: Expandir outline a narrativa completa
- Más robusto pero 2x costo

---

### ❌ BT-4 (HIGH): No hay metric cross-verification — hallucinations invisibles

**Evidencia:** El LLM recibe `facts_context` y `metrics_payload` como JSON, luego genera texto libre. Pero entre la entrada y la salida, no hay verificación de que los números citados coincidan.

**Ejemplo de hallucination típica:**
```
Context: {"accuracy": 0.847, "f1": 0.823}
LLM genera: "El modelo alcanzó una precisión del 91.2% con F1 de 0.89"
→ Ambos números inventados, pero el reporte se entrega al CEO
```

**Esto es especialmente peligroso porque:**
- El CEO no tiene acceso a metrics.json para verificar
- El reporte se presenta como "evidence-driven" y "decision-ready"
- Una decisión de negocio puede basarse en una métrica alucinada

**Fix — `_extract_and_verify_metrics(report_text, facts_context, metrics_payload)` (~60 líneas):**

```python
# Extraer claims numéricos del reporte
patterns = [
    r'(\d+\.?\d*)\s*%',           # porcentajes
    r'[Rr]²?\s*[=:]\s*(\d+\.?\d*)', # R²
    r'[Ff]1\s*[=:]\s*(\d+\.?\d*)',   # F1
    r'accuracy\s*[=:]\s*(\d+\.?\d*)', # accuracy
    r'MAE\s*[=:]\s*(\d+\.?\d*)',      # MAE
    r'RMSE\s*[=:]\s*(\d+\.?\d*)',     # RMSE
]

# Para cada valor extraído, buscar match en contexto (±5% tolerance)
# Si no match → flag como "unverified"
# Si >2 unverified → append warning al reporte
```

---

### ❌ BT-5 (MEDIUM-HIGH): Evidence section es cosmética, no sustantiva

**Evidencia:** Líneas 904-990.

`_build_evidence_items()` genera items genéricos:
```python
items.append({"claim": f"Artifact available: {clean_path}", "source": clean_path})
```

`_ensure_evidence_section()` **REEMPLAZA** la sección de evidencia del LLM con estos items genéricos.

**Problema:** Si el LLM generó buenos evidence items como:
```
{claim: "Accuracy del modelo es 84.7%", source: "data/metrics.json → cv_accuracy_mean"}
```
...se pierden y se reemplazan por:
```
{claim: "Artifact available: data/metrics.json", source: "data/metrics.json"}
```

El evidence tracking pierde toda su utilidad.

**Fix:**
1. Parsear evidence items del LLM primero
2. Validar cada uno (source es path real? claim contiene métrica verificable?)
3. Mantener items válidos del LLM
4. Solo añadir items genéricos para completar mínimo 3
5. Solo reemplazar si LLM no generó la sección o items son inválidos

---

### ❌ BT-6 (MEDIUM-HIGH): PDF generator desconecta imágenes del contexto

**Evidencia:** `pdf_generator.py` líneas 43-46:
```python
images = re.findall(r'!\[(.*?)\]\((.*?)\)', markdown_content)
markdown_text_clean = re.sub(r'!\[(.*?)\]\((.*?)\)', '', markdown_content)
```

Extrae TODAS las imágenes del markdown, las elimina del texto, y las pone en un grid al final del PDF.

**Problema:**
- El translator escribe: "Como muestra la Figura 1 a continuación, los segmentos..."
- En el PDF: la figura está 5 páginas después, en un grid genérico
- El CEO lee "a continuación" → no hay nada a continuación
- Se pierde la conexión narrativa entre insight textual y evidencia visual

**Fix:** Mantener imágenes inline en la conversión HTML:
```python
# En lugar de strip + re-append:
# Convertir ![alt](path) a <img src="abs_path" style="max-width:100%">
# directamente en el HTML, in-place
```

---

### ❌ BT-7 (MEDIUM): Language detection frágil para textos cortos

**Evidencia:** Líneas 26-71.

```python
spanish_markers = {'el', 'la', 'los', 'las', 'un', 'una', ...
    'precio', 'cliente', 'datos', 'modelo', 'resultado',  # ← domain words
    'optimización', 'análisis', 'negocio', 'valor',
}
```

**Problemas:**
1. **Domain words inflados:** "precio", "cliente", "datos" son domain-specific, no stopwords. Un business_objective en inglés sobre pricing: "optimize **precio** of **modelo** X" → detecta español por domain words
2. **Threshold frágil:** `spanish_count > english_count + 2` — con objective de 6 palabras, la diferencia de 2 es ruidosa
3. **Sin fallback a user preference:** Si el usuario configuró idioma en settings, no se usa
4. **Solo es/en:** Sin soporte para pt, fr, de, etc.

**Fix:**
- Eliminar domain words de markers (precio, cliente, datos, modelo, etc.)
- Para textos < 10 palabras: usar default del sistema o pedir al user
- Pasar idioma como parámetro opcional desde graph.py

---

### ❌ BT-8 (MEDIUM): _derive_exec_decision tiene edge cases

**Evidencia:** Líneas 868-889.

```python
if status == "sufficient_signal":
    if lift is None:
        return "GO_WITH_LIMITATIONS"
    return "GO" if lift > 0 else "NO_GO"  # ← lift == 0 → NO_GO
```

- **lift = 0** (modelo no mejora vs baseline) → NO_GO. Pero esto no es "no apto", es "inconcluso". Debería ser GO_WITH_LIMITATIONS.
- **lift = 0.001** (mejora trivial) → GO. Pero un lift del 0.1% no justifica deploy.

Además, líneas 1171-1176:
```python
if outcome in {"GO", "NO_GO", "GO_WITH_LIMITATIONS"}:
    executive_decision_label = outcome  # ← OVERRIDE completo
```

El run_summary.run_outcome **sobreescribe** la derivación cuidadosa. Si run_summary dice "GO" pero data_adequacy dice "insufficient_signal", gana run_summary sin reconciliación.

**Fix:**
- lift == 0 → GO_WITH_LIMITATIONS (no NO_GO)
- lift < threshold configurable (e.g., 0.01) → GO_WITH_LIMITATIONS
- Si run_outcome difiere de derived decision → flag discrepancia para el reporte

---

### ❌ BT-9 (MEDIUM): No hay control de budget de tokens del prompt

**Evidencia:** Línea 1902: `full_prompt = system_prompt + "\n\n" + user_message`

No se mide ni se controla el tamaño total del prompt. Para runs complejas con muchos artifacts, el prompt puede crecer a 50K+ tokens.

**Problemas:**
- Puede exceder context window del modelo
- Tokens desperdiciados en contextos de baja prioridad
- No hay strategy de truncación

**Fix:**
```python
# Antes de enviar:
estimated_tokens = len(full_prompt) // 4  # Approximación
MAX_PROMPT_TOKENS = 28000  # Dejar espacio para respuesta

if estimated_tokens > MAX_PROMPT_TOKENS:
    # Truncar por prioridad:
    # 1. Eliminar contextos duplicados (data_adequacy_context si ya está data_adequacy_report_json)
    # 2. Truncar tablas (scored_sample a 3 rows en vez de 5)
    # 3. Eliminar contextos opcionales (run_timeline, cleaning_context)
```

---

### ❌ BT-10 (MEDIUM): Error fallback es un string plano

**Evidencia:** Línea 1914: `return f"Error generating report: {e}"`

Si el LLM falla, el CEO recibe literalmente "Error generating report: Connection timeout".

**Contraste:** El Strategist genera un "Error Fallback Strategy" completo con analysis_type, difficulty, column_validation.

**Fix — `_generate_deterministic_fallback_report()` (~80 líneas):**
```python
def _generate_deterministic_fallback_report(
    executive_decision_label, facts_context, metrics_payload,
    evidence_paths, error_message
):
    """Generate a minimal but useful report without LLM."""
    sections = []
    sections.append(f"# Executive Report (Auto-generated)\n")
    sections.append(f"## Executive Decision\n{executive_decision_label}\n")
    sections.append(f"*Note: This report was auto-generated due to a translation error. "
                    f"Please review artifacts directly for full details.*\n")
    
    if facts_context:
        sections.append("## Key Metrics\n")
        for fact in facts_context[:5]:
            sections.append(f"- {fact['metric']}: {fact['value']} (source: {fact['source']})")
    
    sections.append(f"\n## Evidence Paths\n")
    for path in evidence_paths[:8]:
        sections.append(f"- {path}")
    
    return "\n".join(sections)
```

---

### ❌ BT-11 (LOW-MEDIUM): sanitize_report_text es agresivo

**Evidencia:** Líneas 892-898:
```python
def _sanitize_report_text(text: str) -> str:
    ...
    while "..." in text:
        text = text.replace("...", ".")
```

Reemplaza **todos** los "..." por "." globalmente. Si el LLM cita un path como `data/model...weights.json` o escribe un rango `0.1...0.9`, se corrompe.

**Fix:** Solo reemplazar "..." al inicio/final de oraciones, no globalmente. O eliminar esta lógica — el prompt ya dice "NO ELLIPSIS".

---

### ❌ BT-12 (LOW): Imports duplicados

**Evidencia:** Líneas 6 y 14:
```python
from typing import Dict, Any, Optional      # línea 6
from typing import Dict, Any, Optional, List  # línea 14
```

No causa error pero indica falta de revisión.

---

## FLUJO INTER-AGENTES: ¿Qué recibe el Translator?

| Fuente | Contenido | Calidad |
|--------|-----------|---------|
| state.execution_output | Output del ML Engineer | ✅ Bueno |
| state.selected_strategy | Estrategia elegida | ✅ Bueno |
| state.review_verdict | Veredicto del Reviewer | ✅ Bueno |
| data/metrics.json | Métricas del modelo | ✅ Bueno |
| data/insights.json | Insights consolidados | ✅ Bueno (si existe) |
| data/run_summary.json | Resumen de ejecución | ✅ Bueno |
| data/data_adequacy_report.json | Evaluación de datos | ✅ Bueno |
| data/alignment_check.json | Alineación con objetivo | ✅ Bueno |
| translator_view | Política de reporting | ✅ Bueno |
| state.plots_local | Paths de visualizaciones | ⚠️ Variable |
| **Feature-target correlations** | **NO EXISTE** | ❌ **No llega** |
| **Compute constraints** | **NO EXISTE** | ❌ **No llega** |
| **Steward signal quality** | **NO PROPAGADO** | ❌ **Se pierde** |

**Gap principal:** Las mejoras P2-1 (feature-target correlations) y P2-2 (STEWARD_FACTS) del Steward deberían propagarse hasta el Translator para que el reporte pueda incluir "Las 5 variables más predictivas son..." respaldado por datos reales.

---

## MEJORAS PRIORIZADAS

### Priority 1 — Quick Wins (1-2h cada una)

**P1-1: Validación estructural post-generación** (~50 líneas)

Añadir `_validate_report_structure(content, executive_decision_label)`:
- Verificar secciones requeridas existen (regex para headers)
- Verificar executive_decision_label aparece en el reporte
- Verificar longitud razonable (500 < chars < 30000)
- Si falla → log warning + append note al reporte

**P1-2: Fallback report determinístico** (~60 líneas)

Reemplazar `return f"Error generating report: {e}"` por `_generate_deterministic_fallback_report()` que produce reporte mínimo pero usable desde facts_context + metrics + evidence_paths.

**P1-3: Fix edge cases en _derive_exec_decision** (~10 líneas)
- lift == 0 → GO_WITH_LIMITATIONS (no NO_GO)
- Si run_outcome difiere de derived → incluir ambos en executive_decision_label con nota

**P1-4: Fix evidence section — preservar items válidos del LLM** (~30 líneas)

Modificar `_ensure_evidence_section()`:
- Parsear items `{claim, source}` del LLM
- Validar cada source (es path real?)
- Mantener válidos, solo añadir genéricos para completar mínimo

**P1-5: Eliminar domain words de language detection** (~5 líneas)

Quitar: 'precio', 'cliente', 'datos', 'modelo', 'resultado', 'optimización', 'análisis', 'negocio', 'valor' de spanish_markers. Y equivalentes de english_markers.

**P1-6: Fix sanitize_report_text** (~5 líneas)

Cambiar `text.replace("...", ".")` por regex que solo afecte "..." precedido/seguido de espacio o fin de línea.

### Priority 2 — Medium Effort (medio día cada una)

**P2-1: Metric cross-verification** (~80 líneas)

`_verify_report_metrics(content, facts_context, metrics_payload)`:
- Extraer claims numéricos del reporte con regex
- Buscar match en facts_context y metrics_payload (±5% tolerance)
- Si >2 no verificables → append warning section
- Si decision-critical metric no verifica → flag para retry

**P2-2: Prompt consolidation — FACTS_BLOCK + NARRATIVE_GUIDE** (~150 líneas refactor)

Reestructurar el prompt en 3 capas:
1. **FACTS_BLOCK** (~2K tokens): Bloque determinístico con hechos duros (decision, metrics, adequacy, review, artifacts). El LLM NO debe modificar estos valores.
2. **NARRATIVE_GUIDE** (~80 líneas): Instrucciones de estilo y estructura. Eliminando redundancias y contradicciones.
3. **CONTEXT_APPENDIX**: Solo contextos que añaden información nueva (no duplicados).

Eliminar contextos redundantes:
- Mantener `data_adequacy_report_json`, eliminar `data_adequacy_context`
- Mantener `model_metrics_context`, eliminar `weights_context` (sobrepuesto)
- Mantener `artifact_manifest_context`, eliminar `artifacts_context` (subconjunto)

**P2-3: Validation + Repair loop** (~100 líneas)

Post-generación:
```
report = generate(prompt)
issues = _validate_report(report, ...)
if issues.has_critical:
    repair_prompt = _build_repair_prompt(report, issues)
    report = generate(repair_prompt)  # 1 retry
    issues = _validate_report(report, ...)
if issues.has_critical:
    report = _deterministic_fallback(...)
    report += "\n\n> ⚠️ Auto-generated due to quality issues"
```

**P2-4: PDF inline images** (~40 líneas en pdf_generator.py)

Cambiar lógica de strip+re-append a inline conversion:
```python
# En lugar de:
markdown_text_clean = re.sub(r'!\[(.*?)\]\((.*?)\)', '', markdown_content)
# Hacer:
def _resolve_inline(match):
    alt, path = match.group(1), match.group(2)
    abs_path = resolve_image_path(path, base_dir_abs)
    if abs_path:
        return f'<img src="{abs_path}" style="max-width:90%" alt="{alt}"/>'
    return f'<em>[Image not found: {alt}]</em>'
html_with_images = re.sub(r'!\[(.*?)\]\((.*?)\)', _resolve_inline, markdown_content)
```

### Priority 3 — Long Term (1+ día)

**P3-1: 2-Pass report generation**

Pass 1 (structured): Generar JSON con outline + hechos clave por sección:
```json
{
  "executive_decision": {"label": "GO_WITH_LIMITATIONS", "reason": "..."},
  "evidence": [{"metric": "accuracy", "value": 0.847, "source": "metrics.json"}],
  "risks": ["small sample (n=450)", "no baseline comparison"],
  "next_actions": ["Add dummy baseline", "Collect 2x more data"]
}
```

Pass 2 (narrative): Expandir JSON a narrativa ejecutiva fluida.

Beneficio: Pass 1 es verificable determinísticamente. Pass 2 solo añade estilo, no puede inventar hechos.

**P3-2: Token budget management**

Medir tokens del prompt, implementar truncación por prioridad con categorías:
- Critical (nunca truncar): FACTS_BLOCK, executive_decision, metrics
- Important (truncar si necesario): sample tables, evidence paths
- Optional (eliminar primero): run_timeline, cleaning_context, slot_payloads

**P3-3: Propagación de signal quality del Steward**

Una vez implementadas las mejoras del Steward (correlaciones, target distribution), propagar al Translator para secciones como "Las variables más influyentes" basadas en datos reales, no en interpretación del LLM.

**P3-4: Report quality scoring**

Implementar `_score_report_quality(content, context)` → 0-100:
- +20 por secciones completas
- +20 por métricas verificadas
- +20 por evidence items con sources válidos
- +20 por coherencia con executive_decision_label
- +20 por ausencia de placeholders/hallucinations
- Si score < 60 → retry o fallback

---

## ASSESSMENT COMPARATIVO

| Dimensión | business_translator | Steward | Strategist | Reviewer |
|-----------|-------------------|---------|------------|----------|
| Líneas de código | 1915 | ~2700 | 1016 | ~800 |
| Context Assembly | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ |
| Prompt Quality | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| Output Validation | ★☆☆☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| Error Recovery | ★☆☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| Determinism | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |

---

## PROYECCIÓN DE MEJORA

| Estado | Score | Descripción |
|--------|-------|-------------|
| **Actual** | **62%** | Fuerte en prep, zero en validación |
| **Con P1** (3-4h) | **74%** | Validación básica + fallback + fixes |
| **Con P1+P2** (2-3 días) | **85%** | Cross-verification + prompt optimizado + repair loop |
| **Con P1+P2+P3** (1 semana) | **92%** | 2-pass + token management + quality scoring |

---

## CONCLUSIÓN

El Business Translator es un agente con **excelente preparación** (context assembly top-tier) pero **zero quality assurance** en su output. Es como un analista que investiga a fondo pero no revisa el informe final antes de enviarlo al CEO.

La debilidad más crítica (BT-1) es que el reporte se entrega sin ninguna verificación de que las métricas citadas son reales, las secciones existen, o la decisión es coherente. Para un agente cuyo output es la **cara visible del sistema ante stakeholders de negocio**, esto es inaceptable.

Las mejoras P1 (3-4 horas) elevan al agente de 62% a 74% — suficiente para operación confiable. Las P2 (2-3 días adicionales) lo llevan a 85% — nivel senior profesional con cross-verification y retry. El ROI más alto está en P1-1 (validación estructural) y P2-1 (metric cross-verification), que juntas previenen el escenario más peligroso: un CEO tomando decisiones basadas en métricas alucinadas.