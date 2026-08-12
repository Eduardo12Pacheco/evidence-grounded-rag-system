# Protocolo de retrieval y evaluación — evidence-grounded-rag-system

Estado: **fórmulas comprometidas, conjunto de consultas comprometido, números
deterministas de fixture sintético.**

## Propósito

Fijar el diseño de retrieval y el protocolo de evaluación para que las
afirmaciones de calidad sean reproducibles en lugar de afirmadas. Cada número a
continuación lo produce un comando contra fixtures comprometidos y un conjunto de
consultas comprometido.

## Diseño de retrieval

### Sparse — Okapi BM25 (`retrieval/bm25.py`)

Puntuación para una consulta `q` y un documento `d`:

    score(q, d) = Σ_{t ∈ q} idf(t) · (tf(t,d) · (k1 + 1)) / (tf(t,d) + k1 · (1 − b + b · |d| / avgdl))

con idf de Robertson-Sparck Jones:

    idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

Parámetros comprometidos: `k1 = 1.5`, `b = 0.75`. Tokenización: `v1` (NFKC,
case-fold, split en ejecuciones no alfanuméricas, **sin stemming** — "review" y
"reviewed" son tokens distintos). Los empates se resuelven por (score desc, id de
fragmento ascendente). Consulta vacía, corpus vacío, `top_k <= 0` y términos sin
coincidencia devuelven todos un resultado vacío.

### Dense — interfaz de embeddings (`embeddings/`)

`EmbeddingProvider` es la interfaz reemplazable (protocolo con `model_id`,
`mode`, `semantic`, `dimension`, `embed`, `embed_one` y `to_dict`). Existen dos
proveedores:

- **`DeterministicEmbedding`** (por defecto) es un vector de bolsa-de-palabras
  con hash (hashing firmado, normalizado L2, dimensión 256) con `semantic = False`,
  `mode = offline`. NUNCA se etiqueta semántico: los puntajes dense codifican solo
  solapamiento de tokens.
- **`SemanticLocalEmbedding`** es opcional, explícito, solo local
  (BAAI/bge-small-en-v1.5, `semantic = True`, `mode = semantic-local`, 384
  dimensiones, pooling CLS, normalizado L2). Carga SOLO desde un caché local
  verificado con `local_files_only=True` y lanza `SemanticModelUnavailableError`
  cuando no está disponible — nunca cae silenciosamente al embedding hash. Ver
  `docs/embedding-model-contract.md`.

El ranking es similitud coseno; los empates se resuelven por (score desc, id de
fragmento ascendente); un vector de consulta cuya dimensión difiere de la
dimensión indexada lanza `EmbeddingDimensionError`; un índice vacío devuelve un
resultado vacío.

### Fusión — fusión de rango recíproco (`retrieval/fusion.py`)

    rrf(chunk) = Σ_{systems s} 1 / (k + rank_s(chunk))

Parámetro comprometido: `k = 60`. Los candidatos se deduplican; los empates se
resuelven por (score fusionado desc, id de fragmento ascendente). Se aplica un
umbral `min_score` (por defecto `0.01`, ver `PipelineConfig`) al ranking
fusionado antes de que la evidencia se exponga al generador. `top_k <= 0` produce
un resultado vacío.

### Modos

`hybrid` (por defecto), `sparse` y `dense` están soportados por el motor híbrido.
En modo `sparse`/`dense` la respuesta se fundamenta en ese ranking único, pero el
trace siempre registra los candidatos de ambos sistemas más el ranking fusionado,
las versiones comprometidas y la configuración.

## Protocolo de evaluación

### Conjunto de consultas (`data/fixtures/queries/synthetic_queries_v1.json`)

Versión `1.0.0`, congelada contra el corpus `synthetic-fixtures` v1.0.0. 11
consultas comprometidas que cubren las siete categorías requeridas:

| Categoría | Consultas | Esperado |
| --- | --- | --- |
| answerable | q-002, q-005, q-006, q-011 | respondida |
| unanswerable | q-009, q-010 | abstenerse `unsupported_query` |
| ambiguous | q-008 | abstenerse `insufficient_evidence` |
| conflicting | q-007 | abstenerse `conflicting_evidence` |
| citation_boundary | q-003 | respondida, cita solo al fragmento expuesto |
| numeric | q-001 | respondida con el número exacto |
| multi_passage | q-004 | respondida con múltiples afirmaciones/citas |

Cada consulta declara `answerable`, `expected_state`, `expected_reason` y juicios
de relevancia graduados (id de fragmento + grado). El loader del conjunto de
consultas rechaza consultas respondidas sin juicios y requiere que cada categoría
esté cubierta.

### Métricas de retrieval (macro-promediadas sobre consultas con ≥1 juicio relevante)

- **Recall@k** (k = 1, 5, 10): `|relevant ∩ top-k| / |relevant|`.
- **MRR**: `1 / rank(first relevant)`, si no 0.
- **nDCG@k** (k = 5, 10): `DCG@k / IDCG@k` con `DCG@k = Σ_{i=1..k} rel_i / log2(i + 1)`.

### Métricas de fidelidad (sobre todos los registros generados)

- **Answerable coverage**: respondidas / consultas respondibles.
- **Abstention rate**: abstenidas (incluidas conflictivas) / todas las consultas
  (descriptiva).
- **Abstention precision**: abstenciones correctas / abstenciones.
- **Abstention recall**: abstenciones correctas / consultas no respondibles. Un
  rechazo es "correcto" cuando la consulta no es respondible Y la razón
  registrada coincide con la razón esperada comprometida (cuando se declara).
- **Unsupported-claim rate**: afirmaciones que fallaron la validación de citas /
  total de afirmaciones.
- **Citation precision**: citas válidas / citas verificadas.
- **Citation completeness**: afirmaciones con al menos una cita válida / total de
  afirmaciones.

### Reglas de validación de citas

Una generación respondida solo se reporta como verificada si cada verificación
pasa; de lo contrario el pipeline rechaza la respuesta (abstención
`citation_validation_failed`) y nunca la expone:

- `citation_not_found` — el fragmento no existe en el corpus servido.
- `citation_not_exposed` — el fragmento no fue parte de la evidencia entregada al
  generador.
- `corpus_version_mismatch` — la cita pertenece a otra versión del corpus.
- `excerpt_mismatch` — el extracto citado difiere del texto del fragmento.
- `invalid_location` — ubicación faltante o no la ubicación `synthetic://` del
  fragmento.
- `claim_without_citation` — una afirmación no abstenida no cita nada.
- `unused_citation` — una cita no es referenciada por ninguna afirmación
  (forjada/huérfana).
- Los rechazos se omiten (`skipped`) por validación (nunca presentados como
  verificados).

### Razones de abstención (obligatorias, códigos estables)

`insufficient_evidence`, `unsupported_query`, `conflicting_evidence`,
`citation_validation_failed`, `provider_unavailable`, `malformed_output`,
`blocked_corpus`, `invalid_corpus`, `stale_corpus`. Los detalles legibles están
comprometidos en `src/evidence_grounded_rag/abstention/reasons.py`.

## Materializar el reporte

```bash
uv run python -m evidence_grounded_rag.evaluation.materialize
```

Opcionales: `--manifest`, `--queries`, `--out`, `--top-k`, `--mode`.

Escribe `data/manifests/evaluation-synthetic-v1.0.0.json` con versiones,
configuración, identidades de modelo, resultados por consulta, métricas, un
timestamp, limitaciones explícitas y las etiquetas `synthetic_evaluation: true`,
`corpus_provenance: project-generated-synthetic`,
`real_corpus_gate_blocked: true`. Las métricas son deterministas; el timestamp es
procedencia registrada.

## Números actuales del fixture sintético (reporte comprometido)

```text
recall@1  0.8333   recall@5  1.0000   recall@10  1.0000
mrr       1.0000   ndcg@5    1.0000   ndcg@10    1.0000
answerable_coverage      1.0000
abstention_rate          0.3636
abstention_precision     1.0000
abstention_recall        1.0000
unsupported_claim_rate   0.0000
citation_precision       1.0000
citation_completeness    1.0000
```

Recall@1 está por debajo de 1 porque las consultas multi-relevantes (q-004,
q-007, q-008) cuentan todos sus fragmentos relevantes en el denominador. Estos
números describen solo los fixtures sintéticos comprometidos y el conjunto de
consultas.

## Reglas de honestidad

- Cada número es reproducible desde un comando y un conjunto de consultas
  comprometido.
- Ninguna métrica se reporta antes de que su fórmula esté comprometida en este
  doc.
- La evaluación con fixture sintético se etiqueta como tal en cada reporte y en
  la demo.
- Sin afirmación de validación del mundo real, métricas perfectas,
  comportamiento libre de alucinaciones, confiabilidad ni aptitud de producción.

## Evaluación local real (separada, diagnóstica)

Una evaluación acotada separada sobre el único corpus local real aprobado
(documentación de Python 3.12.13, 3 archivos / 70 fragmentos). NUNCA se combina
con el reporte sintético: conjunto de consultas separado, comando separado,
archivo de reporte separado, etiquetas separadas.

### Conjunto de consultas real (`data/evaluation/queries/python-docs-real-v1.0.0.json`)

Versión `1.0.0`, congelado contra el corpus `python-docs-3.12.13` v1.0.0 vía el
source manifest `corpus-source-v1.0.0` (revisión `3bb231a6...`). 18 consultas
comprometidas que cubren cada caso requerido:

| Caso | Consultas |
| --- | --- |
| single_passage | real-q-001, real-q-012 |
| multi_passage | real-q-006, real-q-007, real-q-008 |
| exact_api | real-q-004, real-q-010 |
| terminology | real-q-003, real-q-009 |
| numeric_parameter | real-q-002, real-q-011 |
| section_distinction | real-q-005 |
| citation_boundary | real-q-009 |
| unanswerable | real-q-013, real-q-014 |
| ambiguous | real-q-015, real-q-016 |
| abstain | real-q-017, real-q-018 |

Cada consulta declara `query_id`, `case`, `question`, `expected_status`
(`answerable | unanswerable | ambiguous | abstain`), fragmentos gold graduados,
rango esperado de conteo de citas, razón(es) esperada(s) de abstención, notas y
un registro de anotación. El loader (`evaluation/real_queries.py`) valida
esquema, IDs duplicados, id/versión del corpus y referencia del manifest,
existencia de fragmentos en el corpus derivado, cobertura de caso/estado y carga
determinista. El loader de siete categorías sintéticas no se toca.

### Procedimiento de anotación (reproducible)

1. Construir el corpus derivado en memoria desde el source manifest aprobado y
   los archivos locales crudos (`build_real_local_corpus`); los IDs de fragmentos
   son `derive_chunk_id(document_id, seq)`, verificados contra la revisión del
   manifest.
2. Para cada pregunta, localizar la(s) oración(es) que la responden y registrar
   los ID(s) de fragmento derivado(s); grado 2 = primario, 1 = secundario.
3. Redactar la pregunta con las formas exactas de tokens presentes en la oración
   gold porque el generador extractivo comprometido no tiene stemming y requiere
   cada token clave de la consulta en una oración para una coincidencia completa.
4. Declarar el estado esperado a partir del juicio del anotador sobre el
   contenido del corpus, independiente de cualquier modo de retrieval.

### Comando

```bash
uv run python scripts/verify_corpus.py --manifest data/manifests/corpus-source-v1.0.0.json

uv run python -m evidence_grounded_rag.evaluation.real_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --output data/evaluations/python-docs-real-evaluation-v1.0.0.json
```

El comando verifica primero el manifest, la auditoría, los archivos, los tamaños,
el SHA-256 crudo y el escaneo de privacidad, y luego construye el corpus local
real. Ante CUALQUIER fallo sale con código distinto de cero con el código de error
estable y nunca cae al corpus sintético. No hace llamadas de red.

### Reporte (`data/evaluations/python-docs-real-evaluation-v1.0.0.json`)

Registra corpus/fuente/licencia/manifest/revisión/checksums crudos/checksum
canónico/conteos de documentos y fragmentos, conjunto de consultas, configs y
umbrales, métricas por modo, registros por consulta, traces de retrieval, fallos,
timestamp, commit de implementación y limitaciones. Etiquetas requeridas:

- Modo Dense: **`Deterministic hash embedding diagnostic — not semantic retrieval`**
- Modo Hybrid: **`Hybrid diagnostic using BM25 plus deterministic non-semantic hash embeddings`**
- Generación: **`Offline deterministic extractive evaluation — not an LLM result`**
- El reporte declara explícitamente que es una evaluación acotada sobre tres
  archivos de documentación de Python 3.12.13 y 70 fragmentos, que no representa
  toda la documentación de Python, que usa embeddings hash no semánticos,
  generación determinista (no LLM) y que es diagnóstico, no aptitud de
  producción.

En el reporte no se almacena texto a nivel de fragmento ni de afirmación; las
citas llevan solo checksums de extracto y prefijos cortos.

### Métricas (calculadas por modo de retrieval: sparse, dense, hybrid)

Métricas de retrieval (mismas fórmulas que el protocolo sintético,
macro-promediadas sobre consultas con ≥1 juicio gold), reportadas en general y
agrupadas por answerable, single_passage, multi_passage, unanswerable, ambiguous
y abstain:

- Recall@1/5/10, MRR, nDCG@5 (graduado).
- Tasa sin-resultado (consultas cuyo resultado fusionado está vacío después del
  umbral).
- Conteos de hits top-k (consultas cuyo top-k contiene ≥1 fragmento gold).

Métricas de fidelidad sobre las 18 consultas por modo:

- Answerable coverage: respondidas / consultas respondibles.
- Abstention precision: abstenciones correctas / abstenciones.
- Abstention recall: abstenciones correctas / consultas no respondibles (un
  rechazo es correcto cuando la consulta no es respondible Y la razón registrada
  es una de las razones esperadas de abstención declaradas de la consulta).
- Citation precision: citas válidas / citas verificadas.
- Citation completeness: afirmaciones con ≥1 cita válida / total de afirmaciones.
- Unsupported-claim rate: afirmaciones sin cita válida / total de afirmaciones.
- Citation-validation failure rate: generaciones respondidas rechazadas por
  validación / total de consultas.
- Wrong-answer count: consultas respondidas cuyo estado esperado no es
  respondible.

### Política de timestamp y determinismo

`timestamp_utc` es procedencia de runtime y el único campo permitido a diferir
entre ejecuciones. Todos los demás campos son deterministas y están cubiertos por
`deterministic_digest` (SHA-256 de la serialización canónica sin los campos de
timestamp y digest); dos ejecuciones producen porciones deterministas
byte-idénticas. Los puntajes crudos de retrieval se cuantizan a 12 decimales en
el reporte porque la suma en memoria de BM25 agrega contribuciones de términos en
orden hash-aleatorizado de dict/set, que puede mover el último ulp de un float
entre procesos; los rankings (que impulsan cada métrica) no cambian.

### Honestidad para la evaluación local real

- El reporte real es un diagnóstico sobre 3 archivos / 70 fragmentos de
  documentación de Python 3.12.13; NO es una validación de la documentación de
  Python en general.
- El modo Dense es un embedding hash no semántico determinista, no retrieval
  semántico; la generación es extracción determinista, no un LLM.
- Los resultados son diagnósticos, no aptitud de producción.
- Los denominadores son pequeños (12 consultas respondibles / 6 no respondibles
  por modo); una consulta mueve los porcentajes hasta 8,3 puntos.

## Evaluación del materializer semántico (separada, solo retrieval)

Una evaluación de retrieval SEPARADA sobre el mismo corpus local real con el
modelo de embeddings semánticos locales (BAAI/bge-small-en-v1.5). NUNCA se
combina con el reporte sintético ni con el reporte del materializer hash
(`python-docs-real-evaluation-v1.0.0.json`); nunca cae a embeddings hash bajo una
etiqueta semántica. Ver `docs/embedding-model-contract.md` para la auditoría del
modelo, licencia, revisión y política de checksum.

### Adquisición y verificación del modelo (explícita; red solo aquí)

```bash
uv sync --dev --extra demo --extra embeddings
uv run python scripts/acquire_embedding_model.py \
    --manifest data/manifests/embedding-model-v1.0.0.json
```

Descarga la revisión fijada a un directorio de staging, la verifica byte-exacto
contra el manifest comprometido y luego la intercambia atómicamente en
`data/raw/models/bge-small-en-v1.5/` (gitignored); el manifest comprometido nunca
se reescribe (el bootstrap de primera vez requiere el flag explícito
`--bootstrap`).

```bash
uv run python scripts/verify_embedding_model.py \
    --manifest data/manifests/embedding-model-v1.0.0.json
```

Offline: re-verifica el esquema/versión del manifest, id/versión del modelo,
editor, URLs, revisión, licencia, dimensión (384), entrada máxima (512),
dispositivo, metadatos de arquitectura/tokenizador/pooling/normalización, estado,
redistribución, y luego existencia de archivos + tamaños exactos + SHA-256 y
rechaza archivos extra más allá de `expected_files`; cualquier desajuste detiene
la habilitación semántica.

### Comando

```bash
uv run python -m evidence_grounded_rag.evaluation.semantic_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --model-manifest data/manifests/embedding-model-v1.0.0.json \
    --cache-dir data/raw/models/bge-small-en-v1.5 \
    --output data/evaluations/python-docs-real-semantic-evaluation-v1.0.0.json
```

El comando verifica primero el corpus real y luego carga el modelo semántico solo
desde el caché local (`local_files_only=True`). Si el snapshot del modelo no
puede verificarse/cargarse, lanza `SemanticModelUnavailableError` y no escribe
ningún reporte; los modos semánticos se rechazan y nunca degradan al embedding
hash.

### Cuatro modos separados

| Clave de modo | Etiqueta | Qué lo sirve |
| --- | --- | --- |
| `sparse` | `BM25 sparse baseline` | Okapi BM25 comprometido únicamente |
| `hash` | `Deterministic hash embedding diagnostic — not semantic retrieval` | embedding hash no semántico determinista |
| `semantic` | `Local semantic embedding model` | BAAI/bge-small-en-v1.5 dense, solo local |
| `hybrid` | `Hybrid BM25 plus local semantic embeddings` | BM25 + embeddings semánticos locales fusionados con RRF comprometido (`k = 60`) |

Todos los umbrales existentes no cambian: `min_fused_score = 0.01`, `top_k = 10`,
BM25 `k1 = 1.5` / `b = 0.75`, RRF `k = 60`.

### Reporte (`data/evaluations/python-docs-real-semantic-evaluation-v1.0.0.json`)

Solo métricas de retrieval (sin generación, sin LLM). Registra los
hechos/procedencia/licencia/tokenizador/dimensión/normalización/dispositivo/ruta
de caché del modelo y checksums por archivo, hechos del corpus y del conjunto de
consultas, los cuatro modos, métricas de retrieval (mismas fórmulas que el
protocolo real: Recall@1/5/10, MRR, nDCG@5, tasa sin-resultado, conteos de hits
top-k, agrupadas por answerable/single-passage/multi-passage/unanswerable/
ambiguous/abstain), rankings por consulta y rangos de hits, estabilidad de
ranking entre dos ejecuciones en el mismo proceso, observaciones de runtime
(segundos de wall-clock por modo; RSS pico del proceso si se midió; memoria por
modo "unavailable"), timestamp, revisión de código y limitaciones.

Política de determinismo: las métricas, los rankings por consulta, los hechos del
modelo y los configs son deterministas y están cubiertos por `deterministic_digest`;
solo el timestamp y las observaciones de runtime difieren entre ejecuciones (el
runtime es procedencia medida, no contenido). El reporte semántico declara
explícitamente: acotado a 3 archivos / 70 fragmentos, sin afirmación general de
Python, sin LLM, la calidad de la recuperación no implica la fidelidad de la
respuesta y ningún benchmark de la tarjeta del modelo se reclama como métrica de
proyecto.

Ejecute el comando dos veces y compare `deterministic_digest` y
`retrieved_chunk_ids` por consulta para la estabilidad entre ejecuciones; ambas
ejecuciones en el mismo proceso se reportan además bajo `ranking_stability`.

## Evaluación de generación (separada, offline, diagnóstica)

Una evaluación offline SEPARADA del generador extractivo determinista sobre el
mismo corpus local real aprobado y el conjunto de consultas real SIN CAMBIOS
(`python-docs-real-v1`, 18 consultas). Nunca se combina con los reportes
sintético, del materializer hash o del materializer semántico y nunca cae al
corpus sintético. La política de generación está comprometida en
`docs/generation-contract.md` (prompt `grounded-answer-prompt-v1.0.0`, esquema
estructurado `structured-answer-v1`, bloqueador de selección de proveedor,
adaptador en vivo que falla cerrado).

### Comando

```bash
uv run python -m evidence_grounded_rag.evaluation.generation_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/evaluation/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --fixture data/fixtures/answer-evaluation-v1.0.0.json \
    --output data/evaluations/python-docs-real-generation-evaluation-v1.0.0.json
```

El comando verifica primero el corpus real, luego carga el conjunto de consultas
real sin cambios y lo valida cruzadamente contra el fixture de evaluación de
respuestas comprometido (`data/fixtures/answer-evaluation-v1.0.0.json`), que
referencia los 18 IDs de consultas y sus fragmentos gold SIN copiar ningún
pasaje. Luego ejecuta el generador offline determinista con validación estricta
de citas y el evaluador de soporte de afirmaciones, reflejando las reglas
fail-closed del pipeline de servicio. Sin red, sin LLM, sin fallback sintético.

### Reporte (`data/evaluations/python-docs-real-generation-evaluation-v1.0.0.json`)

Registra hechos de corpus/conjunto de consultas/fixture, configs (BM25/RRF/
umbrales sin cambios, `prompt_version`, `schema_version`, regla de soporte de
afirmaciones), métricas, registros por consulta, fallos, timestamp, runtime,
commit de implementación y limitaciones.

Métricas (todas sobre las 18 consultas en el modo evaluado, hybrid por defecto):

- **Answer coverage**: respondidas / consultas respondibles.
- **Abstention precision**: abstenciones correctas / abstenciones.
- **Abstention recall**: abstenciones correctas / consultas no respondibles.
- **Abstention rate** (descriptiva) y **correct abstention count**.
- **Citation precision**: citas válidas / citas verificadas.
- **Citation completeness**: afirmaciones con ≥1 cita válida / total de
  afirmaciones.
- **Citation-validation failure rate**: generaciones respondidas rechazadas por
  validación / total de consultas.
- **Claim support**: conteos de afirmaciones supported / unsupported / unknown y
  `support_rate` (supported / total de afirmaciones) bajo la regla comprometida
  `normalized-exact-containment-v1`.
- **Invalid-output y provider-error counts** (cero para la ejecución offline).
- Campos por consulta: id/caso/estado esperado de consulta, state, razón de
  abstención y detalle truncado, flags answerable/answered/abstained, abstención
  correcta, respuesta incorrecta, ids de fragmentos recuperados y rangos/
  puntajes cuantizados, ids de afirmaciones con veredictos de soporte, checksums/
  prefijos de citas, resumen de validación y procedencia de generación
  (`provider_id`, `model_id`, `mode=offline`, `is_llm_generated=False`).

Etiquetas requeridas: la generación se etiqueta exactamente
**`Offline deterministic extractive evaluation — not an LLM result`**, y la
generación en vivo se etiqueta **`Live generation NOT measured`** porque no hay
proveedor/modelo/credenciales configurados. El reporte no almacena texto de
afirmaciones ni de fragmentos; las citas llevan checksums de extracto y prefijos
cortos.

### Política de timestamp y determinismo

`timestamp_utc` y `runtime_seconds` son procedencia de runtime — los ÚNICOS
campos permitidos a diferir entre ejecuciones. Cada otro campo es determinista y
está cubierto por `deterministic_digest` (SHA-256 de la serialización canónica
sin esas claves); dos ejecuciones producen porciones deterministas byte-idénticas.
Los puntajes crudos de retrieval se cuantizan a 12 decimales (misma política que
el materializer hash).

## Definición de terminado para la evaluación

- [x] Conjuntos de consultas y fórmulas de métricas comprometidos.
- [x] Un comando reproduce todos los números reportados.
- [x] El comportamiento de abstención está testeado, no es opcional.
