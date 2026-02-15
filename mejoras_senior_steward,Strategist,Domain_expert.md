# AUDITORÍA: Steward, Strategist y Domain Expert
## Sesión 5 — Agentes de Planificación Estratégica
**Fecha:** 2026-02-15  
**Scope:** ¿Son agentes senior de un departamento de Data Science?  
**Archivos auditados:** steward.py (1288L), strategist.py (1016L), domain_expert.py (113L), prompts.py (416L), data_atlas.py (388L), data_profile_compact.py (714L), data_profile_preflight.py (297L), senior_protocol.py (73L), context_pack.py (309L), graph.py orchestration (run_steward, run_strategist, run_domain_expert)

---

## 1. STEWARD — Auditor Jefe de Datos

### 1.1 Rol Esperado
Un Data Steward senior debe: (a) hacer una auditoría exhaustiva del CSV que sirva de contexto fiable para todos los agentes downstream, (b) detectar problemas de calidad antes de que causen fallos, (c) establecer la semántica del dataset (target, splits, IDs, training mask).

### 1.2 Fortalezas Detectadas

**✅ Arquitectura de dos pasadas (Pass1 → Evidence → Pass2)**  
- **Pass1** (`decide_semantics_pass1`): Propone hipótesis semánticas (primary_target, split_candidates, id_candidates) y solicita evidence_requests específicos.
- **Evidence Resolution** (`_resolve_steward_evidence_bundle`): Mide missingness, uniques y column_profile directamente del CSV — no del sample.
- **Pass2** (`decide_semantics_pass2`): Finaliza semánticas con evidencia medida: dataset_semantics, dataset_training_mask, column_sets.  
**Veredicto:** Excelente diseño. El LLM propone, la evidencia determinista confirma.

**✅ Composite Sampling (Head + Tail + Random)**  
- Configurable: 40% head, 40% tail, 20% random middle (lines 25-29).
- Logarithmic scaling: 5000 rows para 10MB, 12000 para 1GB (line 47).
- Cap: 15000 rows max. Shuffle post-concat para distribución no sesgada.
- Total rows contados de forma eficiente sin cargar en memoria (`_count_csv_rows`).
- Metadata de sampling transparente propagada a todos los agentes downstream.  
**Veredicto:** Senior-level. Captura distribución temporal y evita sesgo de "solo primeras N filas".

**✅ Dialect Detection robusto**  
- csv.Sniffer + fallback heurístico (count `;` vs `,`).
- Detección de decimal europeo (`,`) vs anglosajón (`.`).
- Resolución automática de conflicto sep==decimal (line 720-731).
- Encoding: intenta utf-8 → latin-1 → cp1252.

**✅ Smart Profiling (`_smart_profile`)**  
- Top 50 columnas con: dtype, n_unique, cardinality tag, null%.
- Ambiguity detection: numeric-looking strings, percent signs, comma decimals, whitespace.
- Name collision detection tras normalización.
- Column glossary con role hints (identifier, date, monetary, flag, etc.).
- PII detection (email, phone, credit card, IBAN) con scrubbing automático.

**✅ `build_dataset_profile` — Profile estructurado completo**  
- type_hints inferidos (numeric/categorical/datetime/boolean).
- missing_frac por columna con `is_effectively_missing_series` (detecta placeholders como "", "NA", "N/A").
- Cardinality con top_values (top 5 per column).
- Numeric summary: mean, std, min, q25, median, q75, max, zero_frac, neg_frac.
- Text summary: avg_len, empty_frac, numeric_like_ratio, datetime_like_ratio.
- Duplicate stats.

**✅ `build_data_profile` — Evidence Layer para Senior Reasoning**  
- Outcome analysis con class_counts (classification) o quantiles (regression).
- Split candidates con unique values evidence.
- Constant columns, high cardinality columns.
- Leakage flags (columns con nombre del outcome).

**✅ Data Atlas + Atlas Summary**  
- Consolidación de todo el profiling en estructura compacta.
- Target/split/ID hints por nombre de columna.
- Column snapshot con type_hint, missing_frac, unique_count.

**✅ Context Quality Validation (`validate_steward_semantics`)**  
- Valida: primary_target exists in headers, training_rows_rule exists, scoring_rows_rule exists.
- Cross-check: si target tiene missingness parcial → training_rows_rule debe mencionarlo.
- Column_sets explicit_columns validados contra headers.
- Quality gate: si `ready=False` → pipeline puede abortar.

**✅ Fallback determinístico si LLM falla**  
- Si el LLM devuelve vacío: summary determinístico con rows, cols, columns, null_frac (line 420-428).
- Evidence bundle siempre determinístico (CSV scanning directo).

### 1.3 Debilidades Críticas

---

**SW-1: NO hay correlación target↔features en el profile (ALTA)**  
**Evidencia:** steward.py lines 1093-1278 (`build_data_profile`), lines 927-1072 (`build_dataset_profile`).  
**Problema:** El Steward genera estadísticas univariadas (missingness, cardinality, quantiles) pero **NUNCA calcula correlaciones entre features y target**. Un Data Steward senior calcularía:  
- Correlación Pearson/Spearman top-20 features vs target (regresión).  
- Mutual information o chi-squared para features categóricos vs target (clasificación).  
- Esto es CRÍTICO porque el Strategist diseña estrategias sin saber qué features tienen señal.  
**Impacto:** El Strategist propone técnicas "a ciegas" sin saber si hay señal predictiva. Puede recomendar modelos complejos cuando no hay correlación, o ignorar features que sí la tienen.  
**Fix:** Añadir a `build_data_profile` un bloque `feature_target_associations` con top-20 correlaciones numéricas + top-10 chi-squared categóricos vs target. ~40 líneas. Coste: O(n * k) donde k = min(50, n_cols).

---

**SW-2: NO hay detección de distribución del target (ALTA)**  
**Evidencia:** steward.py lines 1187-1211 (outcome_analysis).  
**Problema:** Para clasificación calcula class_counts y para regresión calcula quantiles, pero **NO detecta**:  
- Desbalanceo extremo (clase minoritaria < 5%).  
- Skewness en regresión (log-transform needed?).  
- Outliers en el target (IQR method).  
**Impacto:** El Strategist no puede recomendar SMOTE, stratified sampling, o log-transforms porque no tiene la evidencia.  
**Fix:** Añadir a outcome_analysis: `class_imbalance_ratio` (minority/majority), `skewness`, `kurtosis`, `outlier_frac_iqr`. ~25 líneas.

---

**SW-3: NO hay detección de relaciones temporales (MEDIA-ALTA)**  
**Evidencia:** steward.py lines 838-854 (glossary hints), data_atlas.py.  
**Problema:** El Steward detecta columnas con "date"/"time" en el nombre pero **NO verifica si los datos están ordenados temporalmente** ni si hay autocorrelación. Tampoco calcula frecuencia temporal (daily, monthly, yearly).  
**Impacto:** El Strategist puede recomendar random split en datos temporales → leakage. O recomendar cross-validation cuando se necesita time-based split.  
**Fix:** Añadir `temporal_analysis` al profile: detectar columnas datetime, verificar si están sorted, calcular frecuencia, flag `is_time_series`. ~50 líneas.

---

**SW-4: NO hay detección de multicolinealidad (MEDIA)**  
**Evidencia:** steward.py (no existe ningún cálculo de VIF ni correlation matrix entre features).  
**Problema:** Si hay 10 features altamente correlacionadas entre sí, el Strategist no lo sabe. Puede recomendar modelos que sufren multicolinealidad (regresión lineal sin regularización).  
**Impacto:** Estrategias sub-óptimas. El ML Engineer descubre el problema durante ejecución y tiene que improvisar.  
**Fix:** Calcular correlation matrix entre features numéricas, reportar pares con |r| > 0.95. ~30 líneas.

---

**SW-5: Summary LLM-generated = no determinista, no auditable (MEDIA)**  
**Evidencia:** steward.py lines 341-373 (SYSTEM_PROMPT_TEMPLATE → generate_content).  
**Problema:** El summary textual principal es generado por Gemini Flash. Es el artefacto que TODOS los agentes downstream usan como contexto. Pero es: (a) no determinista (varía entre ejecuciones), (b) puede omitir facts críticos, (c) puede inventar interpretaciones, (d) no es verificable contra el profile.  
**Contraste:** El `dataset_profile`, `data_atlas`, `dataset_semantics` SON determinísticos. Pero el `data_summary` textual es el que domina el prompt del Strategist.  
**Impacto:** Si el LLM omite un dato importante del profile en el summary, el Strategist no lo ve.  
**Fix:** Generar un summary determinístico estructurado (STEWARD_FACTS) desde el dataset_profile y data_atlas, y PREPEND al prompt del Strategist como bloque separado del summary LLM. El LLM summary sirve de interpretación, no de fuente de verdad. ~60 líneas.

---

**SW-6: Prompt del LLM no pide información de escala de cómputo (BAJA-MEDIA)**  
**Evidencia:** steward.py lines 341-363 (SYSTEM_PROMPT_TEMPLATE).  
**Problema:** El prompt pide infer dominio, explicar variables, detectar calidad. Pero **NO pide estimar complejidad computacional**: ¿cuánto costará entrenar modelos con N rows × M cols? ¿Caben en memoria? ¿Se necesita sampling para entrenamiento?  
**Impacto:** El Strategist recomienda técnicas sin saber restricciones de cómputo. graph.py tiene `dataset_scale_hints` pero se calcula por separado y es muy básico.  
**Fix:** Incluir en el profile: `compute_hints = {"estimated_memory_mb": rows * cols * 8 / 1e6, "scale_category": "small|medium|large|xlarge", "sampling_recommended": bool}`. El Strategist usa esto para calibrar complejidad.

---

**SW-7: `_smart_profile` solo procesa top 50 columnas (MEDIA para datasets anchos)**  
**Evidencia:** steward.py line 772 (`sorted_cols = all_cols[:50]`).  
**Problema:** Para datasets con 784+ columnas (e.g., MNIST), solo las primeras 50 se analizan en detalle. Las columnas 51-784 no tienen ambiguity check, glossary hints, ni cardinality tags en el summary.  
**Contraste:** `build_dataset_profile` SÍ procesa TODAS las columnas. Pero el summary textual (que el Strategist lee) solo refleja las primeras 50.  
**Fix:** Cambiar a: reportar todas las columnas en el profile determinístico (ya lo hace), y en el summary textual incluir un resumen agregado de las columnas 51+: "Remaining 734 columns: 730 numeric (range [0,255]), 4 categorical".

---

### 1.4 Resumen Steward

| Aspecto | Nivel | Score |
|---------|-------|-------|
| Dialect/encoding detection | Senior | 90% |
| Sampling strategy | Senior | 88% |
| Profiling univariado | Mid-Senior | 78% |
| Semántica (2-pass) | Senior | 85% |
| Correlaciones target↔features | No existe | 0% |
| Distribución/balance del target | Parcial | 45% |
| Detección temporal | Básica | 30% |
| Summary determinístico | Parcial | 55% |
| **GLOBAL STEWARD** | **Mid-Senior** | **68%** |

---

## 2. STRATEGIST — Jefe de Estrategia de Data Science

### 2.1 Rol Esperado
Un Chief Data Strategist senior debe: (a) diseñar estrategias ejecutables basadas en evidencia del data profile, (b) considerar restricciones de cómputo y datos, (c) proponer técnicas calibradas al problema, (d) incluir fallbacks y análisis de viabilidad.

### 2.2 Fortalezas Detectadas

**✅ Prompt extremadamente completo y first-principles (lines 338-542)**  
- 5 pasos de razonamiento: Understand Business → Translate to DS Objective → Context-Aware Design → Dynamic Validation → Evaluate Metrics.
- Exige `objective_reasoning` conectando business goal → objective_type.
- `feasibility_analysis` obligatorio: statistical_power, signal_quality, compute_value_tradeoff.
- `fallback_chain` obligatorio: siempre Plan B y Plan C.
- `expected_lift` cuantificado vs naive baseline.
- Validation strategy data-driven (temporal → time_split, grouped → group_split, imbalanced → stratified).
- Anti-patterns explícitos: "NO use pre-defined metric lists", "NO hardcoded thresholds".

**✅ Wide-schema mode inteligente**  
- Auto-detect si columns > threshold (env: STRATEGIST_WIDE_SCHEMA_THRESHOLD, default 240).
- `_column_families`: detecta prefijo numérico (pixel0, pixel1... pixel783) → resume como family.
- `_build_inventory_payload`: en wide mode envía families + anchors, no lista completa.
- Budget de required_columns configurable (env: STRATEGIST_WIDE_REQUIRED_COLUMNS_MAX, default 48).

**✅ Column validation + repair loop**  
- `_validate_required_columns`: verifica cada columna contra inventory autorizado.
- `_repair_required_columns_with_llm`: si hay columnas inválidas o over budget, repara con LLM + re-valida. Max 3 intentos.
- Scoring: invalid_count * 1000 + over_budget_count → selecciona mejor candidato.
- `_build_required_columns_repair_prompt`: contexto completo para repair.

**✅ `_build_strategy_spec_from_llm` — Strategy Spec generado desde razonamiento LLM**  
- objective_type, metrics, validation, feasibility, fallback_chain, expected_lift extraídos del output LLM.
- Fallback heurístico si LLM no los provee (lines 930-978).
- Leakage risks detectados por keywords en data_summary + user_request.
- Recommended artifacts universales (no hardcoded a problem type).

**✅ JSON repair robusto**  
- `_parse_with_json_repair`: si JSON malformado → LLM repair con contexto completo. Max 2 intentos.
- Preserva draft existente, solo corrige syntax.
- Bounded: trunca a 180K chars para evitar prompt overflow.

**✅ Fallback strategy si API falla**  
- Genera "Error Fallback Strategy" con analysis_type="statistical", difficulty="Low".
- Incluye column_validation y strategy_spec para que el pipeline no crashee.

**✅ SENIOR_STRATEGY_PROTOCOL integrado**  
- Decision Log, Assumptions, Trade-offs, Risk Register.
- "Use dataset scale hints if available; avoid hardcoded thresholds."
- "Provide candidate techniques, then choose one with clear rationale."

### 2.3 Debilidades Críticas

---

**ST-1: Genera 3 estrategias pero solo 1 se usa — desperdicio de tokens (MEDIA)**  
**Evidencia:** strategist.py line 506 ("LIST of 3 objects"), graph.py run_domain_expert selecciona 1.  
**Problema:** El prompt exige 3 estrategias diferentes. El Domain Expert luego elige 1. Generar 3 estrategias completas con required_columns, feasibility_analysis, etc. consume ~3x tokens sin beneficio proporcional. En muchos casos las 3 estrategias son variaciones menores (e.g., "Random Forest" vs "Gradient Boosting" vs "Ensemble of both").  
**Impacto:** Coste LLM triple. Tiempo de generación 2-3x mayor. Risk de JSON malformado aumenta con output más largo.  
**Fix:** Opción A: Generar 3 títulos+hipótesis en Pass1 (compactos), Domain Expert elige, Strategist desarrolla la elegida en Pass2 con full detail. Opción B: Generar 1 estrategia primaria + 1 fallback compacta. Reduce tokens ~50%.

---

**ST-2: NO recibe feature-target correlaciones del Steward (ALTA)**  
**Evidencia:** El data_summary que recibe contiene: data_atlas_summary + dataset_semantics_summary + steward text summary. NINGUNO incluye correlaciones feature↔target (porque el Steward no las calcula - SW-1).  
**Problema:** El Strategist diseña técnicas y selecciona features sin saber cuáles tienen señal predictiva.  
**Impacto:** Puede recomendar "use all 784 columns" cuando solo 50 tienen correlación significativa. O "use XGBoost" cuando la relación es lineal y un modelo simple bastaría.  
**Fix:** Depende de SW-1. Una vez el Steward calcule correlaciones, incluirlas en data_atlas_summary o como bloque separado.

---

**ST-3: NO valida viabilidad de cómputo contra el entorno real (MEDIA-ALTA)**  
**Evidencia:** strategist.py prompt lines 420-443 (FIRST PRINCIPLES FEASIBILITY).  
**Problema:** El prompt PIDE que el LLM evalúe "statistical power" y "compute-value tradeoff", pero le da información genérica: rows, cols, file_size. **NO le dice**: memoria disponible, tiempo máximo de ejecución, si hay GPU, si el sandbox tiene límites.  
**Contraste:** graph.py tiene `_get_heavy_runner_config()` con `script_timeout_seconds` pero esto NO se pasa al Strategist.  
**Impacto:** El Strategist puede recomendar Deep Learning con 500K rows sin saber que el sandbox tiene 15min timeout y 4GB RAM.  
**Fix:** Pasar al Strategist: `compute_constraints = {"max_runtime_seconds": X, "max_memory_mb": Y, "gpu_available": bool, "sandbox_mode": "local|cloudrun"}`. El LLM ajusta complejidad a restricciones reales.

---

**ST-4: Temperature 0.3 = variabilidad innecesaria (BAJA-MEDIA)**  
**Evidencia:** strategist.py line 562 (`temperature=0.3`).  
**Problema:** Para un output JSON estructurado con decisiones determinísticas, temperature 0.3 introduce variabilidad innecesaria. El Steward usa 0.2, el Domain Expert usa 0.1. La estrategia debería ser la más determinista posible dado un mismo input.  
**Fix:** Reducir a 0.1 o usar `response_mime_type: "application/json"` como el Domain Expert.

---

**ST-5: NO hay validación de coherencia interna entre las 3 estrategias (BAJA)**  
**Evidencia:** strategist.py `_normalize_strategist_output` (lines 639-672).  
**Problema:** Solo normaliza formato (dict vs list). No verifica que las 3 estrategias no usen la misma técnica, no repitan columns, o no contradigan entre sí. Resultado: a veces las 3 estrategias son casi idénticas.  
**Fix:** Post-validación: si similarity(strategy_i, strategy_j) > 0.8 (basado en techniques + required_columns), forzar regeneración de una. O simplemente generar 1 (ver ST-1).

---

### 2.4 Resumen Strategist

| Aspecto | Nivel | Score |
|---------|-------|-------|
| Prompt design | Senior+ | 92% |
| First-principles reasoning | Senior | 88% |
| Wide-schema handling | Senior | 85% |
| Column validation + repair | Senior | 87% |
| JSON robustness | Senior | 85% |
| Feature-target awareness | No (depende de Steward) | 20% |
| Compute constraint awareness | Parcial | 35% |
| Strategy diversity | Bajo | 40% |
| **GLOBAL STRATEGIST** | **Mid-Senior** | **72%** |

---

## 3. DOMAIN EXPERT — Selección de Estrategia

### 3.1 Rol Esperado
Un Senior Business Analyst debe: (a) evaluar cada estrategia contra el objetivo de negocio, (b) considerar riesgos y viabilidad, (c) seleccionar la mejor con razonamiento evidenciado, (d) aportar insight de dominio que los técnicos no ven.

### 3.2 Fortalezas Detectadas

**✅ Criterios de evaluación bien definidos**  
- Business Alignment, Technical Feasibility, Implementability, Risk Assessment.
- Bonus explícito (+0.5) para estrategias con fallback plans.
- "Do not penalize complexity by default; evaluate whether complexity is justified."

**✅ Output estructurado JSON con score/reasoning/risks/recommendation**

**✅ response_mime_type: "application/json" = output forzado JSON (line 28)**

**✅ Temperature 0.1 = determinístico**

### 3.3 Debilidades Críticas

---

**DE-1: Solo 113 líneas = agente MÁS débil del sistema (CRÍTICA)**  
**Evidencia:** domain_expert.py — 113 líneas total, de las cuales ~50 son prompt.  
**Problema:** Comparar:  
- Steward: 1288 líneas + 714 data_profile_compact + 388 data_atlas + 297 preflight = ~2700 líneas de lógica.  
- Strategist: 1016 líneas con validation, repair, wide-schema, strategy_spec.  
- Domain Expert: 113 líneas. UNA función (`evaluate_strategies`). Sin validación, sin fallback robusto, sin determinismo.  
**Impacto:** El Domain Expert es el eslabón más débil en la cadena de planificación.

---

**DE-2: ZERO validación determinística — 100% LLM (CRÍTICA)**  
**Evidencia:** domain_expert.py lines 98-108.  
**Problema:** Si el LLM falla → `return {"reviews": []}` → graph.py fallback selecciona primera estrategia sin evaluación. No hay:  
- Score mínimo requerido (una estrategia con score 2/10 puede ganar si las otras fallan).  
- Validación de que los scores son consistentes (score 9 pero risks=["critical data leakage"] debería ser red flag).  
- Verificación de que el review cubre TODAS las estrategias (si LLM solo evalúa 2 de 3).  
- Check de que required_columns de la estrategia ganadora existen en el inventory.  
**Fix:** Añadir `_validate_domain_expert_output()` que: (a) verifica 1 review per strategy, (b) score mínimo 3.0, (c) consistency check score vs risks, (d) fallback scoring determinístico si LLM falla.

---

**DE-3: Selección por matching de título = frágil (ALTA)**  
**Evidencia:** graph.py line 9397 (`next((r for r in reviews if r.get('title') == strat.get('title')), None)`).  
**Problema:** Si el LLM reformula ligeramente el título ("Price Optimization Strategy" → "Optimal Pricing Strategy"), el match falla → score=0 → estrategia ignorada.  
**Fix:** Usar fuzzy matching o matching por índice posicional (review[0] → strategy[0]).

---

**DE-4: NO aporta insight de dominio propio (ALTA)**  
**Evidencia:** domain_expert.py prompt (lines 42-87).  
**Problema:** El prompt dice "Senior Industry Expert" pero **NO recibe información de dominio**. Recibe: data_summary + strategies. No tiene:  
- Knowledge base de best practices por industria.  
- Histórico de runs anteriores (qué funcionó en datos similares).  
- Benchmarks de referencia (¿R²=0.7 es bueno para este tipo de problema?).  
- Contexto sobre restricciones regulatorias o de negocio.  
**Impacto:** El Domain Expert es realmente un "Strategy Scorer", no un "Domain Expert". No aporta conocimiento nuevo.  
**Fix parcial:** Incluir en el prompt: `dataset_memory_context` (histórico de runs similares), y pedirle explícitamente que: (a) identifique riesgos de dominio no mencionados, (b) sugiera técnicas adicionales basadas en el dominio inferido, (c) valide si las métricas propuestas son las correctas para ese dominio.

---

**DE-5: NO evalúa viabilidad de cómputo (MEDIA-ALTA)**  
**Evidencia:** domain_expert.py prompt line 63.  
**Problema:** La evaluación de "Implementability" dice "Consider dataset scale hints from the data summary" pero el data_summary NO incluye compute constraints (ver ST-3). El Domain Expert no puede evaluar si "Deep Learning with 500K rows" es viable en el sandbox disponible.  
**Fix:** Pasar `compute_constraints` al Domain Expert + incluir criterio: "Reject strategies that exceed compute budget."

---

**DE-6: NO hay re-evaluación tras fallos (MEDIA)**  
**Evidencia:** graph.py run_domain_expert retorna `selected_strategy` y nunca se re-invoca.  
**Problema:** Si la estrategia elegida falla durante ejecución (ML Engineer no puede ejecutarla), el pipeline NO vuelve al Domain Expert para elegir la segunda mejor. La restrategización (`restrategize_count`) existe pero va al Strategist, no al Domain Expert.  
**Fix:** En el restrategize flow, si la estrategia falló, pasar el motivo de fallo al Domain Expert junto con las estrategias restantes para que re-evalúe.

---

**DE-7: NO valida que la estrategia elegida sea ejecutable (MEDIA)**  
**Evidencia:** graph.py lines 9386-9421 (selection logic).  
**Problema:** La selección es: max(score). No verifica: (a) ¿required_columns están en el inventory? (b) ¿las técnicas propuestas son conocidas/soportadas? (c) ¿el data_limited_mode aplica y la estrategia lo considera?  
**Fix:** Post-selección: `_validate_selected_strategy()` que verifique coherencia antes de pasar al Execution Planner.

---

### 3.4 Resumen Domain Expert

| Aspecto | Nivel | Score |
|---------|-------|-------|
| Criterios de evaluación | Mid | 65% |
| Output estructurado | Básico | 55% |
| Validación determinística | No existe | 0% |
| Aporte de dominio | No existe | 10% |
| Robustez a fallos LLM | Mínima | 20% |
| Selection logic | Frágil | 35% |
| **GLOBAL DOMAIN EXPERT** | **Junior-Mid** | **32%** |

---

## 4. FLUJO INTER-AGENTES: Steward → Strategist → Domain Expert

### 4.1 ¿Qué recibe el Strategist del Steward?

| Dato | Disponible | Calidad |
|------|-----------|---------|
| data_summary (texto LLM) | ✅ | Variable (LLM-generated) |
| dataset_semantics_summary | ✅ | Buena (determinístico + LLM) |
| data_atlas_summary | ✅ | Buena (determinístico) |
| column_sets_summary | ✅ | Buena (determinístico) |
| column_manifest_summary | ✅ | Buena (determinístico) |
| column_inventory | ✅ | Perfecta (exact CSV headers) |
| context_pack | ✅ | Buena (dialect, artifacts, scale) |
| Feature-target correlations | ❌ | No existe |
| Target distribution details | ❌ Parcial | Solo class_counts/quantiles básicos |
| Temporal analysis | ❌ | No existe |
| Compute constraints | ❌ | No existe |
| Dataset memory (runs previos) | ⚠️ Parcial | Existe en state pero no llega al prompt |

### 4.2 ¿Qué recibe el Domain Expert del Strategist?

| Dato | Disponible | Calidad |
|------|-----------|---------|
| strategies_list (3 strategies) | ✅ | Buena (validated) |
| data_summary (enriched) | ✅ | Variable |
| business_objective | ✅ | Buena (user input) |
| column_validation report | ❌ | No se pasa al Domain Expert |
| strategy_spec | ❌ | No se pasa al Domain Expert |
| compute_constraints | ❌ | No existe |
| dataset_memory_context | ❌ | No se pasa |

### 4.3 Gaps críticos en el flujo

1. **Feature signal no existe** → Strategist diseña "a ciegas" → Domain Expert evalúa sin evidencia de señal.
2. **Compute constraints no fluyen** → Strategies puede ser inviable → Domain Expert no lo detecta → falla en ejecución.
3. **Strategy_spec no llega al Domain Expert** → El Domain Expert evalúa strategies crudas sin el evaluation_plan, feasibility_analysis, etc. que el Strategist ya calculó.

---

## 5. MEJORAS PRIORIZADAS

### Priority 1 — Quick Wins (1-2h cada una)

**P1-1: Domain Expert validation layer (domain_expert.py)**  
Añadir `_validate_reviews()`:
- Verificar 1 review por strategy (por índice, no por título).
- Score mínimo 3.0 para ser seleccionable.
- Si LLM devuelve reviews vacías → scoring determinístico basado en: has_fallback_chain(+2), has_feasibility(+2), required_columns_valid(+3), technique_count(+1), alignment_keyword_match(+2).
- ~50 líneas.

**P1-2: Fuzzy title matching (graph.py run_domain_expert)**  
Cambiar matching de título exacto a matching por índice posicional con fallback fuzzy:
```python
# Primary: positional match
if idx < len(reviews):
    match = reviews[idx]
# Fallback: fuzzy title match
else:
    match = max(reviews, key=lambda r: fuzz.ratio(r.get('title',''), strat.get('title','')))
```
~15 líneas.

**P1-3: Strategy_spec + column_validation al Domain Expert**  
En graph.py run_domain_expert, enriquecer cada strategy con su strategy_spec y column_validation antes de pasar al Domain Expert. El prompt puede incluir: "If column_validation shows invalid columns, penalize -2 points."  
~20 líneas.

**P1-4: Compute hints en el Steward profile**  
En `build_dataset_profile`, añadir:
```python
compute_hints = {
    "estimated_memory_mb": round(n_rows * n_cols * 8 / 1e6, 1),
    "scale_category": "small" if n_rows < 5000 else "medium" if n_rows < 50000 else "large" if n_rows < 500000 else "xlarge",
    "cross_validation_feasible": n_rows < 100000,
    "deep_learning_feasible": n_rows > 10000 and n_cols < 1000,
}
```
~15 líneas. Propagarlo al Strategist via data_atlas_summary.

**P1-5: Target distribution enrichment**  
En `build_data_profile` outcome_analysis, añadir para clasificación: `imbalance_ratio = min_class / max_class`, y para regresión: `skewness`, `kurtosis`.  
~20 líneas.

### Priority 2 — Medio Esfuerzo (medio día cada una)

**P2-1: Feature-target correlations (steward.py)**  
Nueva función `_compute_feature_target_associations(df, target_col, top_k=20)`:
- Numéricas: `df[numeric_cols].corrwith(df[target_col]).abs().nlargest(top_k)`.
- Categóricas: chi-squared test top-10.
- Output: `feature_target_associations: [{col, method, score, direction}]`.
- Incluir en `build_data_profile` y propagar via data_atlas_summary.
- ~80 líneas.

**P2-2: Steward deterministic summary block (steward.py + graph.py)**  
Generar `STEWARD_FACTS` bloque textual determinístico desde dataset_profile:
```
STEWARD_FACTS (deterministic, source-of-truth):
- Shape: 42000 rows × 12 cols
- Target: 'price' (regression, skew=1.8, 0% missing)
- Top correlations: feature_a(0.82), feature_b(0.71), feature_c(0.65)
- Imbalance: N/A (regression)
- Temporal: date column 'fecha' detected, sorted ascending, frequency=daily
- Quality: 3 cols >50% missing, 1 constant, 2 high-cardinality (IDs)
- Compute: ~4MB, scale=small, cross-val feasible
```
Prepend al data_summary antes del LLM summary. ~60 líneas.

**P2-3: Domain Expert restructure (domain_expert.py)**  
Expandir a ~300 líneas:
- `evaluate_strategies()` → recibe strategy_spec + column_validation + compute_hints.
- `_score_deterministic()` → scoring de fallback sin LLM.
- `_validate_selection()` → post-selección: columns exist, técnicas soportadas, compute feasible.
- `_enrich_with_domain_context()` → inyectar dataset_memory_context + domain-specific warnings.
- Prompt expandido: pedir al LLM que identifique riesgos de dominio no técnicos.

**P2-4: Strategist single-strategy mode (strategist.py)**  
Opción configurable (env: STRATEGIST_STRATEGY_COUNT=1|3):
- Si 1: genera 1 estrategia detallada + 1 fallback compacto. Reduce tokens ~60%.
- Si 3: comportamiento actual.
- Domain Expert en modo 1: valida/rechaza la estrategia en lugar de comparar.

### Priority 3 — Largo Plazo (1+ día)

**P3-1: Temporal analysis module (steward.py)**  
Detectar columnas datetime, verificar orden temporal, calcular frecuencia, flag time_series.

**P3-2: Multicollinearity detection (steward.py)**  
Correlation matrix entre features, reportar pares |r| > 0.95.

**P3-3: Domain Expert knowledge base**  
Crear JSON con best practices por dominio inferido (retail, finance, healthcare). El Domain Expert lo consulta para validar estrategias. E.g., "In healthcare, interpretability is required by regulation → penalize black-box models."

**P3-4: Strategy execution preview**  
Antes de pasar al Execution Planner, hacer un "dry run" rápido: verificar que las técnicas propuestas tienen implementaciones disponibles, que las libraries están instaladas, que el timeout es suficiente.

---

## 6. VALORACIÓN GLOBAL

| Agente | Rol Esperado | Nivel Actual | Con P1 | Con P1+P2 |
|--------|-------------|-------------|--------|-----------|
| **Steward** | Auditor Jefe de Datos | Mid-Senior (68%) | 76% | 85% |
| **Strategist** | Chief Data Strategist | Mid-Senior (72%) | 78% | 86% |
| **Domain Expert** | Senior Business Analyst | Junior-Mid (32%) | 52% | 72% |
| **Flujo inter-agentes** | Pipeline de planificación | Mid (58%) | 70% | 82% |

### Conclusión

El **Steward** tiene una arquitectura sólida (2-pass, composite sampling, dialect detection) pero le falta profundidad analítica: no calcula correlaciones, distribuciones detalladas del target, ni análisis temporal. Es un "auditor de metadatos" más que un "auditor de señal".

El **Strategist** tiene el mejor prompt del sistema y buena robustez técnica (wide-schema, repair, validation), pero opera sin información de señal predictiva ni restricciones de cómputo reales. Genera 3 estrategias cuando 1 detallada sería más eficiente.

El **Domain Expert** es el punto más débil: 113 líneas, 0% determinismo, selección frágil por título, sin aporte real de dominio. Es un "scorer de estrategias por LLM", no un "experto de dominio". Necesita la reestructuración más profunda.

Con las mejoras **Priority 1** (3-5 horas de trabajo), el sistema sube de 58% a 70% global. Con **P1+P2** (2-3 días), alcanza 82% — nivel de equipo senior competente.