# Product — evidence-grounded-rag-system

Estado: **implementación completa para el alcance declarado** — evaluación
sintética por defecto y corpus real local aprobado solo para desarrollo; la
generación en vivo con LLM no fue evaluada y Gob.ec permanece bloqueado.

## Plataforma

Web (demo local) + pipeline de evaluación offline.

## Usuarios

- Evaluadores que deben decidir si se puede confiar en una respuesta RAG.
- Investigadores e ingenieros que quieren un benchmark de fidelidad reproducible.
- El autor: demostrar habilidades de retrieval, embeddings y disciplina de
  evidencia.

## Problema

Las respuestas fluidas no son respuestas confiables. Sin cita obligatoria y
abstención, un sistema RAG puede sonar seguro mientras fabrica. Este producto
hace de la evidencia el producto: las respuestas solo son tan buenas como los
pasajes que las respaldan, y el sistema dice "no sé" cuando falta la evidencia.

## Qué NO es todavía

- No es un motor de búsqueda, sin API, sin despliegue de producción.
- No usa ningún corpus real (ninguno está licenciado todavía); solo fixtures
  sintéticos.
- No afirma ningún número de precisión para contenido real — todas las métricas
  comprometidas describen solo fixtures sintéticos.

## Métrica de éxito (producto)

Un revisor puede hacer una pregunta, ver los pasajes citados, verificar cada cita
y observar una abstención limpia — en menos de un minuto, sin leer código.

## Riesgos

- Acceso al corpus: nada se ejecuta sobre documentos reales hasta que las
  licencias estén confirmadas.
- La fidelidad es difícil de medir bien → protocolo comprometido antes de los
  resultados.

## No-objetivos

- Sin SaaS multi-tenant, sin cuentas de usuario.
- Sin aplicación móvil.
- Sin indexación de corpus sin licencia.

## Definición de terminado

- [ ] La demo responde "¿puedo confiar en esta respuesta y por qué?" con citas en
      un minuto.
- [ ] Cada número reportado es reproducible con un comando.
- [ ] La abstención y los límites de cita están testeados y documentados.
