# Demo Script — evidence-grounded-rag-system

Estado: **implementada — demo offline de Streamlit sobre fixtures sintéticos
comprometidos.**

## Qué muestra la demo

Un usuario hace una pregunta y ve, en una sola pantalla:

1. El **panel de corpus/licencia**: id/versión del corpus, procedencia
   (`project-generated-synthetic`), estado de licencia y el bloqueador de corpus
   real exacto (Gob.ec no autorizado; discrepancia de licencia sin resolver).
2. Un banner **SINTÉTICO / OFFLINE** — todo el contenido son datos de fixtures
   ficticios, sin llamadas de red, sin credenciales.
3. **Modos**: hybrid (por defecto), sparse, dense; más un slider de `top_k`.
4. **Resultados sparse / dense / hybrid**: fragmentos clasificados con rango,
   puntaje, id de fragmento y snippet.
5. **Metadatos de pasaje / fuente** para el fragmento de evidencia principal (id
   de documento, título, versión, fecha de captura, fuente, nota de licencia,
   checksum SHA-256, ubicación `synthetic://`).
6. **Respuesta estructurada** con afirmaciones numeradas y sus citas.
7. **Validación de citas**: válida, o los códigos exactos de problemas que fallan
   (la respuesta se rechaza, nunca se muestra como verificada).
8. **Razón de abstención** cuando el sistema se niega (por ejemplo, evidencia
   conflictiva), con el código de razón estable y el detalle.
9. **Versiones y configuración** y el **trace** completo (candidatos
   sparse/dense/fusionados, versiones, configuración) como JSON.

Los fallos son visibles: fallos de gate, fallos de validación y consultas vacías
se renderizan como paneles de error/advertencia explícitos, nunca como pantallas
vacías silenciosas.

## Restricción

La demo se ejecuta SOLO sobre el corpus de fixture sintético comprometido
(`data/fixtures/synthetic_manifest.json`). No realiza fetches en vivo, no usa
credenciales y nunca toca un corpus sin licencia. El gate de corpus real se
muestra como BLOQUEADO.

## El corpus local real NO es parte de la demo

El corpus local real aprobado (documentación de Python 3.12.13,
`approved-for-local-development`) está intencionalmente ausente de la demo. Es un
límite explícito, opt-in, de desarrollo local ejercitado solo por los scripts
offline — la demo, el pipeline por defecto y la evaluación sintética permanecen
sin cambios y solo sintéticos:

```bash
uv run python scripts/acquire_corpus.py
uv run python scripts/verify_corpus.py --manifest data/manifests/corpus-source-v1.0.0.json
```

`acquire_corpus.py` escribe `data/raw/python-docs-3.12.13/` (gitignored) + el
source manifest. `verify_corpus.py` hace la verificación offline de manifest,
revisión, archivos, tamaños, SHA-256 y privacidad.

Ver `docs/corpus-license-contract.md` para la decisión de licencia y por qué
Gob.ec no se usa, y `README.md` para los modos sintético / local real / bloqueado.

## Run book

```bash
uv sync --dev --extra demo
uv run --extra demo streamlit run scripts/demo_retrieval.py
```

Abra la URL local impresa. La aplicación construye el pipeline en memoria desde
el manifest de fixture comprometido (gate + checksum validados en la carga).

## Recorrido (60–90 segundos)

1. **Abra** — el banner SINTÉTICO/OFFLINE y el panel de corpus/licencia son
   inmediatamente visibles, incluido el bloqueador de corpus real.
2. **Consulta soportada** — la consulta por defecto `What must a permit
   application include?` responde con una afirmación citada:
   - Respuesta: "A permit application must include a signed code of conduct and
     a contact phone number. [1]"
   - La afirmación cita `doc-aurora-park-permits::1` en su ubicación
     `synthetic://`.
   - La validación de citas muestra VÁLIDA.
3. **Evidencia conflictiva** — pregunte `How much is the special event fee per
   day?`: el corpus contiene dos tarifas contradictorias, por lo que el sistema
   muestra `CONFLICTING EVIDENCE: conflicting_evidence` y explica la
   contradicción.
4. **Fuera de corpus** — pregunte `What is the fine for littering in
   Millbrook?`: el sistema se abstiene con `unsupported_query`.
5. **Modos** — cambie a sparse o dense y re-ejecute para comparar rankings; la
   respuesta se fundamenta en el modo seleccionado mientras los candidatos de
   ambos sistemas permanecen visibles.
6. **Cierre** — los paneles de trace y versiones/configuración permiten trazar
   cualquier afirmación de vuelta a un fragmento, rango, puntaje y versión
   comprometida específicos.

## Qué la demo NO debe afirmar

- Sin números de precisión reales: las métricas comprometidas describen solo
  fixtures sintéticos y están etiquetadas como tales.
- Sin uso del corpus Gob.ec, jamás, hasta que el contract de corpus/licencia se
  satisfaga.
- Sin redacción de libre-de-alucinaciones, confiable o apta para producción.
