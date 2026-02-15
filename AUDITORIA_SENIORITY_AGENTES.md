# 🔍 AUDITORÍA DE SENIORITY DE AGENTES
## Sistema Multi-Agente de Business Intelligence
**Fecha:** 2026-02-15  
**Run auditado:** `ad3ee87d` (CSV: `business_input.csv`, 900K filas, objetivo clínico)  
**Status final run:** `NEEDS_IMPROVEMENT` → `REJECTED` por gates de seguridad y compliance

---

## 📊 RESUMEN EJECUTIVO

| Componente | Seniority Score (1-10) | Universalidad | Hardcoding | Veredicto |
|---|---|---|---|---|
| **Steward** | 8.0 | ✅ Alta | ⚠️ Bajo | Casi senior |
| **Strategist** | 7.5 | ✅ Alta | ⚠️ Bajo | Sólido pero con margen |
| **Domain Expert** | 6.5 | ⚠️ Media | ⚠️ Medio | Necesita evolución |
| **Execution Planner** | 7.0 | ⚠️ Media-Alta | 🔴 Alto | Cuello de botella crítico |
| **Data Engineer** | 7.5 | ✅ Alta | ⚠️ Bajo | Buen nivel senior |
| **Cleaning Reviewer** | 6.0 | ⚠️ Media | 🔴 Alto | Exceso de reglas hardcodeadas |
| **ML Engineer** | 7.5 | ✅ Alta | ⚠️ Bajo | Buen prompt, frágil en ejecución |
| **Reviewer** | 6.5 | ⚠️ Media | 🔴 Medio-Alto | Demasiado determinístico |
| **QA Reviewer** | 6.0 | ⚠️ Media | 🔴 Alto | Scripting > Razonamiento |
| **Results Advisor** | 7.0 | ✅ Alta | ⚠️ Bajo | Sólido |
| **Review Board** | 6.5 | ⚠️ Media | ⚠️ Medio | Falta profundidad LLM |
| **Business Translator** | 7.0 | ✅ Alta | ⚠️ Bajo | Bien construido |
| **Failure Explainer** | 5.0 | 🔴 Baja | 🔴 Alto | Muy scripteado |
| **graph.py** | 5.5 | 🔴 Baja | 🔴 MUY Alto | Problema sistémico grave |

**Score global del sistema: 6.5/10**

---

## 🏗️ HALLAZGOS POR COMPONENTE

---

### 1. 🧹 STEWARD AGENT (`steward.py` — 1711 líneas)

**Rol:** Primer agente, audita datos, detecta dialecto CSV, genera perfil semántico.

#### ✅ Fortalezas Senior
- **Composite sampling inteligente** (`_read_csv_composite_sample`): Head+Tail+Random para datasets grandes. Esto es pensamiento senior real — entiende que un muestreo solo del head sesga la distribución.
- **Detección de dialecto robusta** (`_detect_csv_dialect`, `_detect_decimal`): usa `csv.Sniffer` + heurísticas como fallback.
- **Smart profiling** (`_smart_profile`): detecta constantes, alta cardinalidad, targets potenciales. No asume columnas.
- **LLM para semántica** (`decide_semantics_pass1`, `decide_semantics_pass2`): usa 2 pasadas LLM para decidir roles semánticos — esto es reasoning, no scripting.

#### ⚠️ Debilidades
- **Tokens hardcodeados para split/temporal** (líneas 1160-1173): `SPLIT_CANDIDATE_TOKENS` y `_TEMPORAL_HINT_TOKENS` son listas estáticas. Un senior real dejaría esta detección al LLM o usaría patrones más genéricos.
- **Warnings de dateutil** repetidos en la run (16 veces): el `_smart_profile` intenta parsear fechas con `pd.to_datetime` sin formato, generando warnings. Falta robustez silenciosa.
- **Tamaño de muestreo fijo** (`_SAMPLE_ROWS = 5000`): debería adaptarse al dataset (ya lo hace parcialmente con `_compute_sample_sizes` pero los ratios son estáticos).

#### 📊 Veredicto: **8.0/10** — Casi senior
El Steward es el agente más cercano a un "senior real". Su lógica de muestreo compuesto y la doble pasada LLM para semántica demuestran pensamiento adaptativo. La principal mejora sería eliminar los token sets estáticos y delegar más detección al LLM.

---

### 2. 📈 STRATEGIST AGENT (`strategist.py` — 1402 líneas)

**Rol:** Genera estrategias de negocio basadas en el data summary y objetivo.

#### ✅ Fortalezas Senior
- **Protocol-driven**: usa `SENIOR_STRATEGY_PROTOCOL` como guía de reasoning.
- **Diversidad de estrategias** (`_ensure_strategy_diversity`): detecta estrategias redundantes usando similaridad de tokens y solicita regeneración. Esto es pensamiento senior.
- **Column families** (`_column_families`): agrupa columnas por prefijo para informar al LLM — facilita razonamiento sobre datasets anchos.
- **JSON repair** con re-prompt al LLM: no falla silenciosamente ante JSON inválido, sino que le pide al LLM que repare.
- **Usa OpenRouter** con fallback models: resiliencia real.

#### ⚠️ Debilidades
- **`_get_wide_schema_threshold` hardcodeado** (default 200): el umbral para "wide dataset" debería derivarse del perfil, no ser un número mágico.
- **`_get_strategy_count` hardcodeado** (default 3): la cantidad de estrategias debería depender de la complejidad del objetivo.
- **Compute constraints estáticas**: los compute_constraints se pasan pero el agente no razona activamente sobre ellos.

#### 📊 Veredicto: **7.5/10** — Sólido
El Strategist demuestra buen razonamiento LLM y mecanismos de calidad. Los umbrales hardcodeados son menores pero indican pensamiento de "script" en vez de "reasoning".

---

### 3. 🎓 DOMAIN EXPERT (`domain_expert.py` — 423 líneas)

**Rol:** Evalúa y puntúa estrategias con criterio de dominio.

#### ✅ Fortalezas Senior
- **Doble capa**: LLM + scoring determinístico como fallback.
- **Validation cruzada** (`_validate_reviews`): compara reviews LLM vs determinísticas y reconcilia.
- **Domain guidance inference** (`infer_domain_guidance`): intenta inferir el dominio desde tokens del objetivo.

#### ⚠️ Debilidades
- **Scoring determinístico demasiado rígido** (`_score_deterministic`): usa pesos hardcodeados para puntuar. Un senior real debería razonar con el LLM y usar el determinístico solo como safety net.
- **`domain_knowledge.py` tiene reglas fijas** para dominios comunes (medical, financial, etc.): esto es anti-universal.
- **Solo 423 líneas**: comparado con otros agentes, su profundidad de razonamiento es limitada.

#### 📊 Veredicto: **6.5/10** — Necesita evolución
El Domain Expert funciona pero su scoring es más "formula" que "razonamiento". Debería apoyarse más en el LLM para evaluación contextual.

---

### 4. 📋 EXECUTION PLANNER (`execution_planner.py` — 10,473 líneas ⚠️)

**Rol:** Genera el Execution Contract V4.1 que gobierna todo el pipeline.

#### ✅ Fortalezas Senior
- **Prompt SENIOR_PLANNER_PROMPT exhaustivo** (417 líneas en `prompts.py`): cubre 15 secciones con reglas de negocio, leakage, data limited mode, gates, etc. Esto es nivel arquitecto.
- **Progressive compilation** con secciones (`prompt_section_core`, `prompt_section_cleaning_contract`): intenta generar el contrato en partes cuando el LLM falla con la generación completa.
- **Contract validation** (`contract_validator.py` — 154K bytes): validación extensiva del contrato generado.
- **Schema registry** para reparación de contratos.

#### 🔴 Debilidades CRÍTICAS
- **ARCHIVO DE 10,473 LÍNEAS**: este es un monolito que mezcla prompt engineering, parsing, validación, fallbacks, y reparación. Un departamento senior tendría esto modularizado.
- **Fallbacks determinísticos que anulan al LLM**: en la run `ad3ee87d`, el planner falló 5 intentos y cayó en `deterministic_scaffold` — esto significa que el LLM NO pudo generar un contrato válido y se usó uno sintético. Esto anula el valor del razonamiento LLM.
- **Token sets hardcodeados masivos** (líneas 55-182): `_DECISIONING_OBJECTIVE_TOKENS`, `_EXPLANATION_REQUIRED_TOKENS`, `_SECONDARY_ANALYSIS_TOKENS` — más de 120 tokens literales en español e inglés para detectar tipo de objetivo. **Esto es lo opuesto a "cero hardcoding".**
- **`CONTRACT_SOURCE_OF_TRUTH_POLICY_V1` estático**: las reglas de precedencia del contrato son un dict literal, no razonamiento.
- **`DOWNSTREAM_CONSUMER_INTERFACE_V1` estático**: define qué consume cada agente como un dict fijo. Un senior derivaría esto del contrato mismo.
- **En la run, generó 27 archivos de prompt/response** en el directorio del planner — indica múltiples reintentos fallidos antes de succeeder con scaffolding.

#### 📊 Veredicto: **7.0/10** — Cuello de botella crítico
El Execution Planner tiene la visión arquitectónica correcta (contract-driven, V4.1, views por agente) pero su implementación es un monolito frágil con demasiados fallbacks que compensan un LLM que no logra generar el output correcto. El prompt es tan largo y prescriptivo que restringe al LLM en vez de empoderar su razonamiento.

---

### 5. 🔧 DATA ENGINEER (`data_engineer.py` — 990 líneas)

**Rol:** Genera script de limpieza de datos driven por contrato.

#### ✅ Fortalezas Senior
- **Contract-driven prompt**: construye contexto desde `de_view`, `contract`, `cleaning_gates`, `runbook` — el script se genera a medida del contrato.
- **Runtime dependency context** (`_build_runtime_dependency_context`): informa al LLM qué paquetes están disponibles. Esto previene imports prohibidos.
- **Selector expansion** (`_build_selector_expansion_context`): expande selectors del contrato a columnas concretas para que el LLM tenga contexto.
- **Code auto-fixes** (`_clean_code`): corrige problemas sintácticos comunes en el output LLM.
- **Repair mode**: puede regenerar scripts con feedback de errores previos.

#### ⚠️ Debilidades
- **En la run `ad3ee87d` el DE generó código con `import sys`** que fue bloqueado por security scan. Esto indica que el prompt del DE no comunica suficientemente las restricciones de sandbox.
- **Static safety scan** (`static_safety_scan.py`) bloquea el import DESPUÉS de generarlo — un senior lo prevendría en el prompt.
- **Code fences en response** detectados ("Warning: code fences detected"): el LLM embebe el código en markdown a pesar de instrucciones contrarias.

#### 📊 Veredicto: **7.5/10** — Buen nivel senior
El DE tiene buen diseño contract-driven. La principal falla es que el LLM aún genera imports prohibidos, lo que sugiere que el prompt necesita enfatizar más las restricciones de sandbox.

---

### 6. 🔍 CLEANING REVIEWER (`cleaning_reviewer.py` — 2868 líneas)

**Rol:** Valida que la limpieza cumple los gates del contrato.

#### ✅ Fortalezas Senior
- **Doble capa** (deterministic + LLM): evalúa gates determinísticamente y luego usa LLM para contexto semántico.
- **Contract-strict rejection**: fuerza REJECTED si faltan `cleaning_gates` en el contrato — fail-fast correcto.
- **Dialect auto-inference**: detecta si el dialecto del archivo limpiado no coincide y re-infiere.

#### 🔴 Debilidades CRÍTICAS
- **`_FALLBACK_CLEANING_GATES` hardcodeados** (líneas 35-81): 6 gates fijos con parámetros literales. Estos deberían venir SOLO del contrato.
- **`_DEFAULT_ID_REGEX` hardcodeado**: regex para detectar columnas ID con strings literales en español ("partida", "invoice").
- **`_DEFAULT_PERCENT_REGEX` hardcodeado**: regex para detectar porcentajes con strings literales ("plazo").
- **Alias map en `_normalize_gate_name`** (líneas 689-707): mapeo manual de variantes de nombres de gates. Un senior derivaría esto con fuzzy matching o normalization semántica.
- **2868 líneas**: demasiado código determinístico para lo que debería ser una evaluación predominantemente LLM.

#### 📊 Veredicto: **6.0/10** — Exceso de reglas hardcodeadas
Este agente ejemplifica el anti-patrón principal del sistema: compensar la debilidad del LLM con más código determinístico, creando un ciclo donde el agente se vuelve menos adaptable.

---

### 7. 🤖 ML ENGINEER (`ml_engineer.py` — 2968 líneas)

**Rol:** Genera código ML end-to-end basado en el contrato.

#### ✅ Fortalezas Senior
- **System prompt excepcional**: `SYSTEM_PROMPT_TEMPLATE` es uno de los mejores prompts del sistema — define modos (BUILD/REPAIR), precedencia de fuentes, hard constraints, preflight gates, dtype safety, y output safety de forma declarativa.
- **CONTRACT_EXECUTION_MAP obligatorio**: antes de entrenar, el código generado debe imprimir un mapa de ejecución. Esto es auditoría de engineering level.
- **PREFLIGHT GATES** (A-E): gates check explícitos antes del `fit()`. Diseño senior.
- **ML Plan generation** (`generate_ml_plan`): genera plan de entrenamiento separado antes del código real.
- **Iteration handoff normalization**: estructura la información entre iteraciones.

#### ⚠️ Debilidades
- **En la run `ad3ee87d`, generó `import sys`**: misma falla que el DE. El LLM ignora las restricciones de sandbox a pesar de estar en el prompt.
- **Solo 1 iteración antes del abort**: el código fue rechazado por security y no hubo re-intento exitoso del ML (la segunda iteración cayó al DE de nuevo y el DE también falló con security).
- **`REQUIRED_PLAN_KEYS` hardcodeado** (líneas 39-47): claves del ML plan como lista literal.
- **`DEFAULT_PLAN` hardcodeado** (líneas 48-88): plan de fallback con valores placeholder.

#### 📊 Veredicto: **7.5/10** — Buen prompt, frágil en ejecución
El ML Engineer tiene el mejor prompt system del sistema, pero la realidad de la ejecución muestra que los LLMs aún hacen imports prohibidos. La solución no es más script, sino mejor comunicación LLM.

---

### 8. 📝 REVIEWER (`reviewer.py` — 877 líneas)

**Rol:** Review de código ML y evaluación de resultados.

#### ✅ Fortalezas Senior
- **Two-phase evaluation** (`evaluate_results`): Phase 1 deterministic triage, Phase 2 LLM semantic.
- **Deterministic prechecks** (`_deterministic_reviewer_prechecks`): verifica target columns en AST antes de LLM.
- **Reviewer gate filter**: aplica solo gates relevantes del contrato.

#### ⚠️ Debilidades
- **`_deterministic_diagnostics_blockers` hardcodeado** (líneas 284-340): lógica de bloqueo con strings literales para detectar tipos de error.
- **Fallback determinístico rígido** (`_deterministic_eval_fallback`): cuando el LLM no está disponible, usa reglas fijas que no razonan sobre el contexto.
- **No usa `senior_protocol`** más allá de importar `SENIOR_EVIDENCE_RULE`.

#### 📊 Veredicto: **6.5/10** — Demasiado determinístico
El Reviewer debería confiar más en el LLM para evaluación semántica. Su capa determinística es demasiado prescriptiva.

---

### 9. 🛡️ QA REVIEWER (`qa_reviewer.py` — 1270 líneas)

**Rol:** Gate de calidad estricto sobre el código generado.

#### ✅ Fortalezas Senior
- **Análisis AST profundo**: detecta synthetic data, data leakage, split fabrication, y random calls analizando el AST del código generado.
- **Security gate** (`security_sandbox`): verifica imports prohibidos determinísticamente.
- **Target variance guard**: verifica que el target tenga varianza suficiente.

#### 🔴 Debilidades CRÍTICAS
- **`_REGRESSOR_KEYWORDS` hardcodeado** (líneas 615-636): lista literal de nombres de regresores. No es universal.
- **`CONTRACT_BROKEN_FALLBACK_GATES` hardcodeado** (líneas 50-55): gates de emergencia fijos.
- **Detección de leakage basada en strings**: `_is_random_call`, `_is_split_fabrication_call` usan pattern matching en nombres de funciones, no razonamiento semántico.
- **1270 líneas de lógica determinística** con mínimo LLM: este agente es más un "linter" que un "senior QA".

#### 📊 Veredicto: **6.0/10** — Scripting > Razonamiento
El QA Reviewer funciona como una herramienta estática, no como un agente senior que razona. Debería usar el LLM para evaluación de calidad contextual y reservar el AST scanning para gates de seguridad hard.

---

### 10. 💡 RESULTS ADVISOR (`results_advisor.py` — 1011 líneas)

**Rol:** Genera insights y recomendaciones a partir de resultados.

#### ✅ Fortalezas Senior
- **Deployment recommendation** (`_compute_deployment_recommendation`): razona sobre suficiencia de datos y métricas para recomendar deployment.
- **Leakage audit** (`_extract_leakage_audit`, `_feedback_indicates_leakage_risk`): detecta señales de leakage de manera multi-fuente.
- **Plateau detection** (`_detect_plateau`): identifica estancamiento en metrics history.
- **Iteration recommendation**: sugiere cambios entre iteraciones basado en contexto actual.

#### ⚠️ Debilidades
- **Métricas y umbrales hardcodeados** en `_compute_deployment_recommendation` (`min_rows=200`).
- **`_objective_metric_priority`** con prioridades fijas por tipo de objetivo.

#### 📊 Veredicto: **7.0/10** — Sólido
Buen balance entre análisis determinístico e insights LLM. Necesita menos umbrales fijos.

---

### 11. ⚖️ REVIEW BOARD (`review_board.py` — 284 líneas)

**Rol:** Adjudicador final que consolida outputs de reviewers.

#### ✅ Fortalezas Senior
- **Conflict reconciliation** (`_apply_conflict_reconciliation`): resuelve conflictos entre Reviewer y QA.
- **Progress policy** (`_apply_progress_policy`): decide si una iteración tiene suficiente progreso para aprobarse.
- **Deterministic fallback** bien estructurado.

#### ⚠️ Debilidades
- **Solo 284 líneas**: para ser el "adjudicador final", es muy ligero. Un board senior haría análisis más profundo.
- **El fallback determinístico** (`_fallback`) toma decisiones con heurísticas simples cuando el LLM falla.
- **No tiene memory de iteraciones previas** para evaluar tendencias.

#### 📊 Veredicto: **6.5/10** — Falta profundidad
Necesita más razonamiento LLM y contexto histórico para hacer adjudicaciones verdaderamente senior.

---

### 12. 📊 BUSINESS TRANSLATOR (`business_translator.py` — 2553 líneas)

**Rol:** Genera reporte ejecutivo con datos del proceso.

#### ✅ Fortalezas Senior
- **Language detection** (`_detect_primary_language`): auto-detecta español/inglés en el contenido.
- **Artifact manifest exhaustivo** (`_build_report_artifact_manifest`): inventaría todos los artefactos producidos.
- **Table rendering** (HTML y ASCII): soporte dual para web y PDF.
- **KPI snapshot, compliance table, timeline** — report structure profesional.

#### ⚠️ Debilidades
- **2553 líneas de formatting code**: la mayor parte es logística de rendering, no razonamiento LLM.
- **Templates de tabla hardcodeados**: styles inline en HTML.

#### 📊 Veredicto: **7.0/10** — Bien construido
Funciona bien como generador de reportes. La parte LLM (traducción ejecutiva) es sólida.

---

### 13. 🔥 FAILURE EXPLAINER (`failure_explainer.py` — 178 líneas)

**Rol:** Explica errores runtime para informar reintentos.

#### 🔴 Debilidades CRÍTICAS
- **Fallback completamente hardcodeado** (`_fallback`, líneas 143-167): if/elif chain con 12 patrones de error literales. **Esto es exactamente lo que buscamos eliminar.**
  ```python
  if "list of cases must be same length" in lower:
      return "np.select called with mismatched conditions..."
  if "numpy.bool_" in lower and "not serializable" in lower:
      return "json.dumps failed because numpy.bool_..."
  ```
- **Prompts aceptables** para DE y ML failure explanation, pero el fallback anula su valor cuando el LLM no está disponible.

#### 📊 Veredicto: **5.0/10** — Muy scripteado
Este agente debería confiar 100% en el LLM para explicación. El fallback debería ser un "No se pudo diagnosticar automáticamente" en vez de reglas pattern-matching.

---

### 14. 🔗 GRAPH.PY (`graph.py` — 21,154 líneas ⚠️⚠️⚠️)

**Rol:** Orquestador principal del pipeline.

#### 🔴 PROBLEMA SISTÉMICO: MONOLITO DE 21K LÍNEAS

Este es el **problema #1 del sistema**. Un archivo de 950KB con 404 funciones es inmantenible e imposible de razonar coherentemente.

#### 🔴 Hallazgos Críticos de Anti-Seniority

1. **El graph.py hace el trabajo de los agentes**: funciones como `_build_cleaned_data_summary_min` (líneas 2433-2630), `_build_signal_summary_context` (líneas 2243-2397), `_build_required_sample_context` (líneas 2019-2091) — son 400+ líneas de preparación de datos que deberían estar en los agentes o en utils.

2. **Duplicación de lógica**: `_resolve_required_outputs`, `_resolve_expected_output_paths`, `_resolve_contract_columns`, `_resolve_contract_columns_for_cleaning`, `_resolve_allowed_columns_for_gate` — 5 funciones que hacen variaciones del mismo trabajo de resolución de columnas.

3. **Exceso de "glue code" determinístico**: 
   - `_apply_static_autofixes` (líneas 1434-1545): 100+ líneas de regex fixes sobre código generado.
   - `_harmonize_review_packets_with_final_eval` (líneas 1076-1156): reconciliación manual de reviews.
   - `_apply_review_consistency_guard` (líneas 1026-1073): más guard rails manuales.
   
4. **Funciones que deberían ser agentes independientes**:
   - `run_execution_planner` (12166-12584 = **418 líneas**): orquesta al planner con retry infinito.
   - `run_data_engineer` (12625-15555 = **2930 líneas**): ¡Esto es más grande que el agente DE mismo!
   - `run_engineer` (15611-16322 = **711 líneas**): más código ML que el propio agente.
   - `execute_code` (17048-18306 = **1258 líneas**): sandbox execution + cleanup.
   - `run_result_evaluator` (18586-19784 = **1198 líneas**): evaluación post-ejecución.

5. **El graph.py toma decisiones que deberían ser de los agentes**:
   - Selección de columnas requeridas.
   - Aplicación de autofixes estáticos al código.
   - Determinación de si el output contract es válido.
   - Reconciliación de métricas y reviews.

6. **En la run `ad3ee87d`**: el flow fue:
   ```
   Steward ✅ → Strategist ✅ → Domain Expert ✅ → 
   Planner ⚠️ (5 intentos, fallback scaffold) → DE ✅ → 
   Cleaning Review → ML ❌ (import sys bloqueado) → 
   Reviewer → QA → Review Board → Retry → 
   Planner (otra vez, 5 intentos, scaffold) → DE ❌ (import sys otra vez) → 
   Translator (sin resultados) → PDF
   ```
   
   **Observación crítica**: El sistema re-ejecutó todo el pipeline desde el Planner en la segunda iteración. Un equipo senior real NO repetiría la estrategia ni la limpieza de datos — arreglaría SOLO el código ML que falló.

---

## 🎯 DIAGNÓSTICO DE LA RUN FALLIDA

### Root Cause Analysis

1. **El ML Engineer generó `import sys`** → bloqueado por `static_safety_scan.py`
2. **El prompt del ML Engineer SÍ lista restricciones** pero de forma genérica ("Avoid network/shell operations") sin enumerar explícitamente `sys` como prohibido.
3. **El retry loop en graph.py re-ejecuta desde el planner** en vez de solo regenerar el código ML con feedback específico.
4. **En la segunda iteración, el DE TAMBIÉN generó `import sys`** → doble fallo.
5. **El sistema generó reporte PDF sin resultados** — un senior abordaría o re-intentaría antes de reportar vacío.

### Lo que haría un equipo senior real:
1. El ML Engineer recibiría feedback: "Tu código fue rechazado porque usaste `import sys`. Aquí están los imports permitidos: [lista]. Regenera SOLO el código ML."
2. NO se repetiría la estrategia, el contrato, ni la limpieza de datos.
3. Si el ML falla 3 veces, el Review Board escalaría con diagnóstico específico.

---

## 🔴 ANTI-PATRONES DETECTADOS

### 1. **"Compensation Scripting"** (Anti-patrón #1)
Cuando el LLM falla, en vez de mejorar el prompt o la comunicación, se añade más código determinístico para compensar. Esto crea un ciclo donde:
- Más código → prompts más largos → LLM más confundido → más fallbacks → más código.
- **Evidencia**: `graph.py` creció a 21K líneas, `execution_planner.py` a 10K, `cleaning_reviewer.py` a 2.8K.

### 2. **"Token Set Snowball"** (Anti-patrón #2)
Listas crecientes de tokens hardcodeados para detectar patrones que el LLM debería inferir:
- `_DECISIONING_OBJECTIVE_TOKENS` (~40 tokens)
- `_EXPLANATION_REQUIRED_TOKENS` (~20 tokens)
- `_SECONDARY_ANALYSIS_TOKENS` (~30 tokens)
- `SPLIT_CANDIDATE_TOKENS` en steward
- `_REGRESSOR_KEYWORDS` en QA reviewer

### 3. **"God Orchestrator"** (Anti-patrón #3)
`graph.py` no es un orquestador — es un mega-agente que hace el trabajo de todos:
- Prepara contexto para los agentes (debería hacerlo cada agente).
- Aplica autofixes sobre código (debería hacerlo el agente que generó el código).
- Reconcilia reviews (debería hacerlo el Review Board).
- Resuelve columnas y outputs (debería hacerlo el Execution Planner).

### 4. **"Retry Amnesia"** (Anti-patrón #4)
El loop de retry re-ejecuta desde el planner, perdiendo todo el contexto de la iteración anterior. Un equipo senior pasaría:
- Exactamente QUÉ falló (gate específico y evidencia).
- QUÉ estaba bien (no tocar lo que funciona).
- Feedback acumulativo de iteraciones previas.

### 5. **"Fallback Override"** (Anti-patrón #5)
Los fallbacks determinísticos anulan las decisiones LLM en vez de complementarlas:
- `_FALLBACK_CLEANING_GATES` se inyectan aunque el contrato tenga otros gates.
- `deterministic_scaffold` reemplaza completamente el contrato LLM.
- `_deterministic_eval_fallback` en Reviewer ignora contexto LLM previo.

---

## 📋 ESTADO DE COMUNICACIÓN INTER-AGENTE

| Comunicación | Estado | Comentario |
|---|---|---|
| Steward → Strategist | ✅ Buena | data_summary + dataset_profile fluye bien |
| Strategist → Domain Expert | ✅ Buena | strategies + column sets |
| Domain Expert → Planner | ⚠️ Media | critique viaja pero no siempre se usa |
| Planner → DE | ✅ Buena | contract + de_view |
| Planner → ML | ✅ Buena | contract + ml_view |
| DE → Cleaning Review | ✅ Buena | cleaned_data + manifest + cleaning_view |
| ML → Reviewer | ⚠️ Media | code + output, pero falta contexto de ejecución |
| ML → QA | ⚠️ Media | code, pero falta iteration_handoff detallado |
| Reviewer + QA → Board | ⚠️ Media | packets pero reconciliación es manual en graph.py |
| Board → Retry | 🔴 Pobre | retry_handler no pasa diagnóstico completo |
| Todo → Translator | ✅ Buena | artifacts + run_facts_pack bien integrados |

**Principal gap de comunicación**: La iteración retry. El feedback del Board al siguiente intento ML/DE es insuficiente y pierde contexto.

---

## 🏆 RECOMENDACIONES PRIORIZADAS

### P0 — Críticas para universalidad

1. **Desacoplar graph.py**: Mover `run_data_engineer` (2900 líneas), `run_engineer` (711), `execute_code` (1258), `run_result_evaluator` (1198) a archivos separados en `src/graph/steps/`. Graph.py debería ser solo la definición del grafo y routing.

2. **Eliminar token sets hardcodeados**: Reemplazar `_DECISIONING_OBJECTIVE_TOKENS`, `_EXPLANATION_REQUIRED_TOKENS`, etc. con clasificación LLM en el Planner. Las detecciones de tipo de objetivo deberían ser responsabilidad del Strategist/Planner via LLM.

3. **Mejorar retry loop**: El retry debe pasar al ML Engineer exactamente:
   - El error specific (no "NEEDS_IMPROVEMENT")
   - Los imports prohibidos (lista explícita)
   - Los outputs que faltan
   - Lo que SÍ funcionó (no empezar de cero)

### P1 — Importantes para seniority

4. **Eliminar `_FALLBACK_CLEANING_GATES`**: Los gates deben venir SOLO del contrato. Si el contrato no tiene gates, el planner debe regenerar, no el reviewer inventar gates.

5. **Reducir Execution Planner prompt**: 417 líneas de prompt son demasiado prescriptivas. Un senior necesita directrices, no un manual de 15 secciones con ejemplos literales. Reducir a ~100 líneas de principios + dejar que el LLM razone.

6. **Failure Explainer → LLM-only**: Eliminar el `_fallback` de if/elif chain. Si no hay LLM, devolver diagnóstico vacío.

7. **QA Reviewer → Más LLM**: Mantener security_sandbox y leakage detection como AST checks, pero delegar quality evaluation al LLM con evidencia.

### P2 — Mejoras de calidad

8. **Import allowlist en prompts de DE y ML**: Los prompts deben listar explícitamente `import sys` como PROHIBIDO junto con la allowlist de imports permitidos.

9. **Domain Expert → LLM-primary scoring**: El scoring determinístico debería ser solo para cuando el LLM falla, no para "validar" al LLM.

10. **Review Board → Context history**: El Board debería recibir métricas de iteraciones previas para detectar progreso/estancamiento.

---

## 📊 CONCLUSIÓN

El sistema tiene una **arquitectura sólida** (contract-driven, V4.1, views por agente, pipeline con retry) pero sufre de **"compensation scripting"** donde cada falla LLM se "arregla" con más código determinístico, creando un sistema cada vez más rígido y menos universal.

**El gap principal es**: `graph.py` (21K líneas) actúa como un mega-agente que toma decisiones por los agentes reales, reduciendo su autonomía y su capacidad de razonamiento LLM. 

**Para llegar a un equipo senior de data science real**: la lógica de decisión debe migrar de `graph.py` y los fallbacks determinísticos hacia los prompts y el razonamiento LLM de cada agente, reservando el código determinístico SOLO para gates de seguridad hard (imports prohibidos, filesystem safety) y plumbing técnico (lectura de archivos, serialización JSON).

**Score actual: 6.5/10 → Target: 8.5/10**

La diferencia entre 6.5 y 8.5 es: **confiar en el razonamiento LLM como fuente primaria de decisión, con código determinístico solo como safety net, no como driver.**
