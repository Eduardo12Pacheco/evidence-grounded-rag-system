# AGENTS.md — evidence-grounded-rag-system

## Objetivo

Construir recuperación híbrida (embeddings sparse + dense), generación basada en
evidencia con citas, abstención cuando la evidencia es insuficiente y evaluación
de fidelidad. Proyecto independiente, greenfield. NO debe copiar el prototipo
BM25 existente y NO debe presentar el corpus Gob.ec como autorizado.

## Estado

Implementación completa para el alcance declarado, con evaluaciones sintéticas
y evaluaciones reales acotadas documentadas.

## Límites

- NO modificar estos proyectos hermanos: `ecuador-job-market-intelligence`,
  `ecuador-mobility-reliability`, `demand-inventory-optimizer`,
  `ecuador-public-information-evidence-assistant`, `eduardo-github-profile`.
- La publicación actual ya está configurada en GitHub; no crear nuevos remotes
  ni hacer push sin autorización explícita del usuario.
- El corpus local de documentación de Python está auditado para desarrollo
  local. Gob.ec permanece bloqueado (ver `docs/corpus-license-contract.md`).

## Estructura

```text
src/evidence_grounded_rag/     # paquete bajo layout src
tests/                         # pytest; solo tests reales, sin cobertura falsa
docs/                          # contract de corpus/licencia, protocolo de retrieval y evaluación, demo script
data/fixtures/                 # fixtures versionados pequeños (nunca un corpus real)
data/manifests/                # manifests versionados de artefactos capturados
data/raw/ data/processed/      # salida de ejecución gitignored
deploy/                        # notas de despliegue (más adelante)
```

## Comandos esperados

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run --extra demo streamlit run scripts/demo_retrieval.py
```

## Política de datos

- El corpus Gob.ec permanece bloqueado y NO se reclama como autorizado.
- Nunca comprometer `data/raw/` ni `data/processed/`.
- Comprometer únicamente fixtures y manifests pequeños.

## CodeGraph

Usar el índice `.codegraph/` para consultas estructurales. Nunca comprometer su
contenido.

## Testing

Los tests deben verificar el comportamiento real que existe. Sin tests
placeholder que pretendan que las características existen.
