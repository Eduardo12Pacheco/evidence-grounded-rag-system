# Contract del modelo de embeddings — BAAI/bge-small-en-v1.5 (local, opt-in)

Estado: **`approved-for-local-development`** — revisión fijada `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, adquirido localmente, nunca comprometido, nunca redistribuido.

## Ruta rápida

Paso 1: instalar el stack de dependencias opcionales (torch/transformers son un
gran footprint).

```bash
uv sync --dev --extra demo --extra embeddings
```

Paso 2: adquirir el modelo fijado en el caché gitignored (el ÚNICO paso de red).

```bash
uv run python scripts/acquire_embedding_model.py --manifest data/manifests/embedding-model-v1.0.0.json
```

La adquisición normal trata el **manifest comprometido como el ancla de
confianza**: carga y audita el manifest comprometido, descarga exactamente los
archivos fijados del manifest a un directorio de staging temporal, los verifica
byte-exacto contra los tamaños/SHA-256 declarados del manifest **antes** de tocar
el caché, y luego renombra atómicamente el directorio de staging verificado sobre
el caché. Una ejecución fallida o parcial nunca puede dejar un caché aparentemente
válido, y el manifest comprometido **nunca se reescribe** — una descarga
comprometida no puede auto-certificarse sobrescribiendo los checksums
comprometidos. El bootstrap de primera vez (sin manifest todavía) es una acción
explícita y separada:

```bash
uv run python scripts/acquire_embedding_model.py --bootstrap \
    --manifest data/manifests/embedding-model-v1.0.0.json
```

El bootstrap es el único modo que escribe un manifest nuevo, requiere que el
manifest esté ausente y sus checksums están auto-certificados (no existía un ancla
de confianza): el comando imprime una advertencia clara de que el manifest nuevo
debe revisarse y comprometerse antes de tratarse como aprobado.

Paso 3: verificar el snapshot local byte-exacto (offline, sin descargas).

```bash
uv run python scripts/verify_embedding_model.py --manifest data/manifests/embedding-model-v1.0.0.json
```

Paso 4: ejecutar la evaluación separada del materializer semántico.

```bash
uv run python -m evidence_grounded_rag.evaluation.semantic_materialize \
    --manifest data/manifests/corpus-source-v1.0.0.json \
    --queries data/queries/python-docs-real-v1.0.0.json \
    --raw-dir data/raw/python-docs-3.12.13 \
    --model-manifest data/manifests/embedding-model-v1.0.0.json \
    --cache-dir data/raw/models/bge-small-en-v1.5 \
    --output data/evaluations/python-docs-real-semantic-evaluation-v1.0.0.json
```

El modelo semántico es **opt-in y explícito**. El embedding por defecto sigue
siendo el embedding hash determinista, no semántico (`semantic = False`, sin
pesos, sin red). **No hay fallback silencioso**: si el modelo local o su snapshot
verificado no está disponible, la recuperación semántica lanza
`SemanticModelUnavailableError` y se rechaza.

## Auditoría de candidatos (registrada 2026-08-11)

Los seis candidatos son páginas oficiales de modelo de Hugging Face. La selección
es reproducible desde evidencia oficial; las razones de rechazo se registran para
que la decisión no deba re-derivarse.

| Candidato (página oficial) | ¿Rechazado? | Razón |
| --- | --- | --- |
| **BAAI/bge-small-en-v1.5** | **SELECCIONADO** | Orientado a retrieval en inglés, con licencia MIT, 384-dim, entrada de 512 tokens, sin prefijo de consulta obligatorio para v1.5, compacto y factible en CPU, público y sin gate. |
| sentence-transformers/all-MiniLM-L6-v2 | Rechazado | La truncación de 256 tokens entra en conflicto con los fragmentos de 2000 caracteres del corpus (cortaría la evidencia silenciosamente). |
| sentence-transformers/all-mpnet-base-v2 | Rechazado | Footprint mayor (~420 MB) del necesario para una evaluación acotada de 70 fragmentos. |
| Snowflake/snowflake-arctic-embed-xs | Rechazado | El prefijo de consulta obligatorio (`query:`) agrega un contract de preprocesamiento frágil. |
| intfloat/multilingual-e5-small | Rechazado | Convenciones de prefijo obligatorias (`query:`/`passage:`) y pesos multilingües más pesados; el corpus actual es inglés. |
| nomic-ai/nomic-embed-text-v1 | Rechazado | Ruta de código remoto personalizada `trust_remote_code` — más frágil y difícil de fijar que un modelo BERT WordPiece plano. |

Ningún candidato se afirma como universalmente mejor, y ningún número de
benchmark de la tarjeta del modelo se reporta jamás como métrica de proyecto.

## Modelo seleccionado — evidencia oficial

| Campo | Valor |
| --- | --- |
| Id del modelo | `BAAI/bge-small-en-v1.5` |
| Editor | Beijing Academy of Artificial Intelligence (BAAI) |
| Página oficial del modelo | https://huggingface.co/BAAI/bge-small-en-v1.5 |
| Tarjeta oficial del modelo (cruda) | https://huggingface.co/BAAI/bge-small-en-v1.5/raw/main/README.md |
| API / configuración oficial | https://huggingface.co/api/models/BAAI/bge-small-en-v1.5/revision/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| Revisión fijada (SHA del commit) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Tag de revisión | ninguno (fijado por SHA de commit; nunca una rama mutable) |
| Licencia | MIT License (`license: mit` en el YAML de la tarjeta; "FlagEmbedding is licensed under the MIT License. The released models can be used for commercial purposes free of charge.") |
| Texto de la licencia | https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE |
| Fecha de adquisición | 2026-08-11 |
| Estado del modelo | `approved-for-local-development` |

## Hechos del modelo (desde los archivos de configuración fijados)

| Hecho | Valor | Fuente |
| --- | --- | --- |
| Arquitectura | BERT, `BertModel`, 12 capas, 12 cabezas | `config.json` |
| Dimensión del embedding | **384** (`hidden_size` = 384) | `config.json`, `1_Pooling/config.json` |
| Entrada máxima | **512** tokens (`max_seq_length` = 512, `max_position_embeddings` = 512) | `sentence_bert_config.json`, `config.json` |
| Pooling | **CLS** — `1_Pooling/config.json` define `pooling_mode_cls_token: true`; el README oficial instruye "Perform pooling. In this case, cls pooling. `sentence_embeddings = model_output[0][:, 0]`" | `1_Pooling/config.json`, README oficial |
| Normalización | **L2** — README oficial: "normalize embeddings `torch.nn.functional.normalize(..., p=2, dim=1)`"; la cadena de módulos de sentence-transformers agrega un módulo `2_Normalize` | README oficial, `modules.json` |
| Tokenizador | WordPiece, `bert-base-uncased` (vocab 30522, lowercasing), `special_tokens_map.json` `[CLS] [MASK] [PAD] [SEP] [UNK]` | `tokenizer.json`, `vocab.txt`, `config.json` |
| Formato de pesos | un único `model.safetensors` (float32) | árbol de archivos del repo |
| Dispositivo | CPU (un solo hilo para inferencia determinista) | nuestro contract de proveedor |

Nota de pooling: el README oficial de BGE muestra pooling CLS para uso directo de
`transformers`. El wrapper de sentence-transformers gestiona el pooling y la
normalización a través de su propia cadena `modules.json` (Transformer →
`1_Pooling` CLS → `2_Normalize`); la salida del wrapper es el vector normalizado
L2 de 384-dim.

## Política de uso permitido y redistribución

- La licencia MIT permite la reproducción, análisis, testing, exhibición y uso
  derivado locales; retenida para el desarrollo local de este repositorio.
- **La política de redistribución es `not-redistributed`** (ver
  `data/manifests/embedding-model-v1.0.0.json`): los pesos se descargan una vez al
  caché gitignored `data/raw/models/bge-small-en-v1.5/` y **nunca se comprometen,
  empaquetan ni publican** por este repositorio.
- El artefacto comprometido es el manifest pequeño
  (`data/manifests/embedding-model-v1.0.0.json`) que contiene metadatos y los
  **checksums SHA-256 reales y tamaños de bytes** de cada archivo adquirido —
  nunca los pesos.
- Los artefactos de terceros conservan sus propios términos (MIT para este
  modelo); la licencia MIT de este repo cubre solo el código original y los
  fixtures sintéticos.

## Política de checksum

- Cada archivo adquirido se registra en el manifest con su tamaño byte-exacto y
  SHA-256 crudo (`sha256_bytes` sobre los bytes descargados exactos).
- `model.safetensors` se contrastó con el hash de blob LFS del Hub
  `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`.
- La verificación (`scripts/verify_embedding_model.py`) re-verifica
  esquema/versión, id/versión del modelo, editor, URLs, revisión, licencia,
  dimensión, entrada máxima, dispositivo, metadatos de
  arquitectura/tokenizador/pooling/normalización, estado, redistribución, y luego
  existencia de archivos + tamaño exacto + SHA-256 exacto, y rechaza cualquier
  archivo extra en el caché más allá de los `expected_files` del manifest.
  Cualquier desajuste detiene la habilitación semántica.

## Comportamiento de inferencia

- **Solo local**: `SentenceTransformer(local_path, device="cpu",
  local_files_only=True)`. Sin red, sin acceso al Hub, sin revisión de fallback.
- **Configuración CPU determinista**: un solo hilo (`torch.set_num_threads(1)`),
  `batch_size=16` fijo, orden de entrada estable, inferencia float32, L2 vía el
  módulo `2_Normalize` del modelo más `encode(normalize_embeddings=True)`.
- **Verificación de dimensión**: la dimensión del embedding del modelo cargado se
  verifica igual a la 384 comprometida antes de cualquier puntuación; un desajuste
  lanza `SemanticModelUnavailableError`.
- **Entrada vacía**: `embed([])` devuelve `[]`.
- **Entrada larga**: las entradas se truncan por el tokenizador a 512 tokens,
  exactamente como se comporta el modelo oficial; documentado, no silencioso.

## Memoria esperada (aproximada, cualitativa)

Solo cualitativa — sin mediciones inventadas. BGE-small es un BERT de 12 capas y
384 de ancho (~130 MB de pesos float32). En CPU, el modelo más el tokenizador más
un índice de 70 fragmentos se espera que quepa cómodamente en unos pocos cientos
de MB de RAM; el índice dense de 70 fragmentos en sí es despreciable comparado
con el modelo cargado. La evaluación registra el RSS pico del proceso que observó
realmente (ver `runtime_observations` del reporte); no se afirma ninguna cifra de
memoria por modo.

## Footprint de dependencia e instalación

- `sentence-transformers>=3.0` detrás del extra opcional
  `[project.optional-dependencies].embeddings` (ecosistema con licencia
  MIT/Apache).
- Instalarlo arrastra el stack de PyTorch/Transformers (torch, transformers,
  tokenizers, safetensors, scikit-learn, numpy), que es un gran footprint de
  disco/instalación. Por eso es opcional y nunca requerido por el pipeline
  central, la demo por defecto ni la suite de tests.
- Los tests nunca descargan pesos; la inferencia real se ejercita solo cuando la
  dependencia opcional y el snapshot local verificado están presentes.

## Comportamiento no disponible (explícito, sin fallback)

- **El proveedor aplica el límite él mismo.** `SemanticLocalEmbedding` requiere un
  manifest auditado: la construcción audita el manifest contra las constantes de
  aprobación y verifica por bytes el caché (existencia, tamaño exacto, SHA-256
  exacto y **sin archivos extra más allá de los `expected_files` del manifest**)
  antes de cargar cualquier cosa. Los metadatos provistos por el llamador nunca se
  confían y no se usa una lista de archivos críticos hardcodeada.
- Caché de modelo faltante o incompleto / cualquier archivo faltante, alterado o
  extra → `SemanticModelUnavailableError` listando el bloqueador exacto.
- `sentence-transformers` no instalado → `SemanticModelUnavailableError` con el
  comando de instalación.
- `local_files_only=False` o un dispositivo no aprobado → rechazado
  (`SemanticModelUnavailableError`); `local_files_only=True` y CPU son
  invariantes.
- Dimensión del modelo cargado desconocida o ≠ 384 → `SemanticModelUnavailableError`
  (la dimensión nunca se omite silenciosamente).
- Fallos de esquema/política de auditoría del manifest (manifest faltante, modelo
  no aprobado, desajuste de revisión/licencia/metadatos, estado bloqueado) →
  `EmbeddingModelManifestError` con un código estable; estos se propagan sin
  envolver.
- En cada caso el modo semántico se rechaza y **nunca** degrada al embedding hash
  determinista bajo una etiqueta semántica.

## Limitaciones

- Este es un contract de evaluación acotada para un corpus de 3 archivos / 70
  fragmentos; no valida la documentación de Python en general.
- La calidad de la recuperación no implica la fidelidad de la respuesta; el
  materializer semántico reporta solo métricas de recuperación (sin generación,
  sin LLM).
- Ningún modelo se afirma como universalmente mejor; los benchmarks de la tarjeta
  del modelo no son métricas de proyecto.
