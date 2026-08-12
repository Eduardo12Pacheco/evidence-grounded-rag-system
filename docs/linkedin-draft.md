# Borrador LinkedIn — sistema RAG basado en evidencia

> Publicación profesional en español (español neutro). Largo objetivo: lectura
> de ~60-90 segundos. Los placeholders de GitHub/demo se reemplazan antes de
> publicar.

---

Cuando un sistema RAG responde con fluidez, no por eso responde con verdad.

Construí un sistema de generación aumentada por recuperación (RAG) donde cada
respuesta está ligada a evidencia: recuperación híbrida (BM25 + embeddings
densos fusionados por RRF), generación con citas por fragmento, abstención
obligatoria y evaluación de fidelidad reproducible.

Tres decisiones que importan:

1. Cada afirmación cita el fragmento exacto que la respalda. La validación de
   citas es estricta y a prueba de fallos: si una cita no existe, no fue
   expuesta al generador o no coincide con el fragmento, la respuesta se
   rechaza; nunca se muestra como verificada.
2. Abstención obligatoria con código de razón estable. Cuando la evidencia es
   insuficiente o contradictoria, el sistema se abstiene formalmente con un
   código de razón estable y explica el motivo, en lugar de inventar.
3. Evaluación reproducible de retrieval y soporte de afirmaciones con métricas y
   conjuntos de consultas versionados: cada número se reproduce con un comando,
   sin sorpresas ni métricas improvisadas.

Un hallazgo de confiabilidad que me parece central: la calidad de la
recuperación no demuestra la fidelidad de la respuesta. Por eso se reportan por
separado. Además, el soporte de una afirmación se mide como contención
normalizada (el texto de la afirmación está textualmente en el fragmento
citado), no como entailment semántico.

Alcance real, sin exagerar:
- La evaluación por defecto usa fixtures sintéticos versionados (contenido
  ficticio, no datos reales).
- Un corpus real local aprobado (documentación oficial de Python 3.12.13, 3
  archivos / 70 fragmentos) se usa solo para desarrollo local.
- Los embeddings semánticos locales (BAAI/bge-small-en-v1.5) son opcionales y
  nunca sustituyen silenciosamente a los deterministas.
- La generación con LLM en vivo NO fue evaluada: no hay proveedor ni
  credenciales configuradas. La generación offline es extractiva y
  determinística, no un LLM.
- Un corpus con licencia en disputa (Gob.ec) fue descartado y no se usa.

Proyecto en progreso, documentado como tal: contratos de licencia, protocolos y
reportes de evaluación comprometidos en el repositorio.

GitHub: [placeholder-repo]
Demo local: [placeholder-demo]
