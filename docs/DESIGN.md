# Design — evidence-grounded-rag-system

Estado: **implementación completa para el alcance declarado** — evaluación
sintética por defecto y corpus real local aprobado solo para desarrollo; la
generación en vivo con LLM no fue evaluada y Gob.ec permanece bloqueado.

## Problema

Los sistemas RAG se juzgan por la fluidez percibida, no por si la respuesta está
respaldada. Este proyecto construye un pipeline RAG donde cada respuesta está
ligada a la evidencia: la recuperación híbrida expone pasajes, la generación los
cita, la abstención se niega a fabricar y la evaluación de fidelidad mide la
brecha entre afirmación y evidencia.

## Arquitectura

```text
src/evidence_grounded_rag/
├── corpus/        # carga de corpus con gate de licencia, IDs, checksums, normalización
├── text/          # tokenizador determinista compartido (usado por retrieval/embeddings)
├── retrieval/     # BM25 + dense + fusión RRF + motor híbrido y traces
├── embeddings/    # interfaz de embeddings + embedding determinista no semántico
├── generation/    # modelos de afirmación/cita, proveedor extractivo, adaptador de entorno
├── validation/    # validación estricta de citas (fail-closed)
├── abstention/    # códigos de razón de abstención obligatoria
├── evaluation/    # métricas comprometidas, conjunto de consultas versionado, comando materialize
└── pipeline.py    # EvidenceRagPipeline de extremo a extremo (retrieve → generate → validate)
```

Cada capa está detrás de una interfaz documentada y es reemplazable.

## Decisiones clave (implementadas)

| Tema | Decisión |
| --- | --- |
| Lenguaje | Python >=3.11, `uv`, layout src de hatchling; stdlib para el núcleo |
| Recuperación sparse | Okapi BM25, `k1 = 1.5`, `b = 0.75`, idf de Robertson-Sparck Jones, tokenizador determinista (`v1`, sin stemming) |
| Recuperación dense | Protocolo `EmbeddingProvider`; el `DeterministicEmbedding` incluido es un vector de bolsa-de-palabras con hash con `semantic = False` (offline, sin pesos, nunca etiquetado como semántico) |
| Fusión | Fusión de rango recíproco, `k = 60` (comprometido; cambiarlo cambia cada número) |
| Generación | Proveedor extractivo/template offline determinista (`is_llm_generated = False`); el adaptador configurado por entorno falla cerrado a `provider_unavailable` para cualquier proveedor no offline |
| Abstención | Comportamiento requerido, testeado; cada rechazo lleva un código de razón estable |
| Evidencia | Las afirmaciones se mapean solo a pasajes recuperados; el generador nunca ve el corpus completo |
| Citas | Validación estricta: existe, expuesta, versión actual del corpus, extracto exacto, ubicación estable, referenciada por una afirmación; las respuestas no verificadas nunca se exponen |
| Gate de corpus | Falla cerrado: corpus reales bloqueados, licencias no aprobadas, errores de checksum/formato/duplicado y versiones obsoletas se niegan a servir |
| Evaluación | Fórmulas comprometidas + conjunto de consultas versionado comprometido; un comando reproduce todos los números |
| Demo | Aplicación Streamlit local sobre fixtures comprometidos únicamente |

## Procedimiento de decisión del generador extractivo

El generador recibe solo los fragmentos de evidencia recuperados y aplica un
procedimiento comprometido y completamente determinista:

1. Evidencia vacía → abstenerse `insufficient_evidence`.
2. Extraer afirmaciones coincidentes (oraciones que contienen un token clave de
   la consulta). Ninguna → abstenerse `unsupported_query`.
3. Conflicto: entre afirmaciones que coinciden completamente con los tokens
   clave de la consulta (cayendo a todas las afirmaciones coincidentes cuando
   ninguna lo hace), dos afirmaciones de fragmentos distintos con el mismo
   predicado normalizado pero valores escalares diferentes (o negación
   explícita) → `conflicting_evidence`.
4. Consulta multi-aspecto (marcador de coordinación o dos palabras wh) →
   respondida con las afirmaciones coincidentes principales.
5. De lo contrario (un solo aspecto): coincidencias completas de fragmentos
   distintos con sujetos diferentes pero valores idénticos → ambiguo →
   `insufficient_evidence`; si no, respondida con la coincidencia completa
   principal (o la afirmación principal cuando todas las afirmaciones
   coincidentes comparten un predicado).

Las reglas de conflicto/ambigüedad son deliberadamente estrechas y están
documentadas en `src/evidence_grounded_rag/generation/extractive.py`; no son
comprensión semántica.

## No-objetivos

- Sin servicio de búsqueda de producción ni despliegue público.
- Sin uso de corpus sin licencia, jamás (Gob.ec no está autorizado).
- Sin copia del código del prototipo BM25 existente.
- Sin afirmaciones de paridad de benchmark, validación del mundo real, métricas
  perfectas, comportamiento libre de alucinaciones, confiabilidad ni aptitud de
  producción.

## Riesgos

- Licencia del corpus sin resolver → la capa de corpus real permanece bloqueada
  por diseño.
- Alucinación → mitigada por verificación de citas, abstención obligatoria y
  validación fail-closed (las respuestas no verificadas nunca se exponen).
- Juego de métricas → protocolos fijos, fórmulas comprometidas y conjuntos de
  consultas comprometidos.

## Definición de terminado (para el proyecto, no solo el hito actual)

- [x] Capa de corpus con gate de licencia sobre fixtures sintéticos (corpus real
      bloqueado).
- [x] Recuperación híbrida (BM25 + dense determinista + RRF) implementada y
      testeada.
- [x] Cita, abstención y validación estricta implementadas y testeadas.
- [x] Evaluación determinista sobre el conjunto de consultas comprometido,
      reproducida por un comando.
- [x] Demo se ejecuta desde fixtures sin acceso a red.
- [x] Sin afirmaciones sin respaldo en README ni docs.
- [ ] Contract de licencia del corpus real satisfecho para cualquier corpus real
      (bloqueado).
