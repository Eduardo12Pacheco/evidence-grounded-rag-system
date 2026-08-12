# Contract de corpus y licencia — evidence-grounded-rag-system

Estado: **fixtures sintéticos servidos; un corpus local real aprobado solo para
desarrollo local (documentación de Python 3.12.13). Todos los demás corpus
reales permanecen BLOQUEADOS.**

## Propósito

Definir las condiciones bajo las cuales se puede usar CUALQUIER corpus de
documentos, registrar los gates implementados que las aplican en código y
registrar la única excepción aprobada. Este es un gate duro: **ningún corpus real
se usa hasta que su licencia esté confirmada y registrada aquí.** El corpus
Gob.ec está explícitamente NO autorizado y NO se presenta como tal.

Existen dos límites separados, no intercambiables:

1. **Gate de fixture sintético** (`corpus/license_gate.py` + `corpus/loader.py`) —
   sirve el corpus de fixtures ficticio comprometido. Sin cambios.
2. **Auditoría de source manifest** (`corpus/audit.py`, `corpus/real_local.py`,
   `corpus/privacy.py`) — un límite separado, explícito y opt-in para el único
   corpus local real aprobado. Nunca afloja el gate sintético, nunca alimenta la
   demo por defecto y nunca alimenta la evaluación sintética.

## Decisión — el único corpus aprobado (registrado 2026-08-11)

**La documentación oficial de Python 3.12.13 está aprobada solo para desarrollo
local** (`status: approved-for-local-development`). El software de Python y su
documentación se publican bajo la **Python Software Foundation License Version
2**, que otorga reproducción, análisis, testing, exhibición, obras derivadas,
distribución y uso, requiriendo que se retengan el acuerdo de licencia y el
aviso de copyright y que los derivados distribuidos resuman los cambios
realizados.

| Propiedad | Valor |
| --- | --- |
| Id / versión del corpus | `python-docs-3.12.13` / `1.0.0` |
| Editor | Python Software Foundation |
| Docs oficiales | https://docs.python.org/3.12/ |
| Licencia | Python Software Foundation License Version 2 |
| URL de licencia (docs) | https://docs.python.org/3.12/license.html |
| URL de licencia (PSF) | https://www.python.org/psf/license/ |
| Fuente | https://github.com/python/cpython/tree/v3.12.13/Doc |
| Revisión fijada | commit `3bb231a6a5dc02b95658877318bf61501a7209e9` (tag `v3.12.13`) |
| Fecha de acceso | 2026-08-11 |
| Archivos capturados | `Doc/tutorial/introduction.rst`, `Doc/library/functions.rst`, `Doc/reference/lexical_analysis.rst` |
| Estado de auditoría | `approved-for-local-development` |
| Redistribución | `not-redistributed` — la captura cruda es solo local, nunca se compromete |
| Auditoría de privacidad | `no-private-content-detected` |

### Evidencia de licencia (PSF License Version 2, citada de la página oficial)

> Section 2: "PSF hereby grants Licensee a nonexclusive, royalty-free,
> world-wide license to reproduce, analyze, test, perform and/or display
> publicly, prepare derivative works, distribute, and otherwise use Python
> alone or in any derivative version, provided, however, that PSF's License
> Agreement and PSF's notice of copyright ... are retained in Python alone or
> in any derivative version prepared by Licensee."
>
> Section 3: "In the event Licensee prepares a derivative work that is based
> on or incorporates Python or any part thereof, and wants to make the
> derivative work available to others ... Licensee hereby agrees to include in
> any such work a brief summary of the changes made to Python."

### Por qué la redistribución NO se reclama aquí

La licencia PSF permite la redistribución con avisos retenidos y un resumen de
cambios. Esta unidad de trabajo adopta deliberadamente el estado más estrecho
`approved-for-local-development`: los archivos crudos capturados permanecen en el
árbol `data/raw/` gitignored y nunca se comprometen, empaquetan ni publican. Si
una unidad de trabajo futura quiere redistribuir, debe elevar el estado a
`approved-for-redistribution` con una `redistribution_justification` explícita.

## Candidatos considerados y rechazados

| Candidato | Razón de rechazo |
| --- | --- |
| Rust Book | Complejidad de `main` mutable / scope de edición — la fuente canónica se mueve entre ediciones y la revisión "actual" no es un único snapshot fijado estable. |
| Sitio de Kubernetes | El alcance de licencia/procedencia no es suficientemente estable para esta unidad de trabajo — el sitio agrega muchos contribuyentes y repos con licencias por página; sin un par limpio de revisión fijada + licencia. |
| Corpus Gob.ec | Explícitamente no autorizado (ver más abajo); no es un candidato. |

## El bloqueador de corpus real (exacto, actual)

Cada corpus EXCEPTO la captura local aprobada `python-docs-3.12.13` permanece
bloqueado. El corpus Gob.ec está explícitamente no autorizado: su etiqueta
"Creative Commons Attribution 4.0 International" entra en conflicto con los
términos `by/3.0/ec` enlazados, y la discrepancia de licencia no está resuelta.

Este texto está comprometido como `REAL_CORPUS_BLOCKER` en
`src/evidence_grounded_rag/corpus/license_gate.py`, expuesto textualmente en
`corpus_status()` del pipeline y mostrado en la demo.

## Gate obligatorio 1 — fixtures sintéticos (aplicado en código, sin cambios)

El gate vive en `src/evidence_grounded_rag/corpus/license_gate.py` y lo aplica
`src/evidence_grounded_rag/corpus/loader.py` en cada carga. FALLA CERRADO:

- Un manifest que declara `real_corpus: true` se rechaza con el código de error
  `blocked_real_corpus` a menos que el gate registre una licencia confirmada de
  forma independiente. Hoy nunca lo hace — cada corpus real está bloqueado por
  este gate.
- Un manifest cuyo estado de licencia no es `approved` se rechaza con el código
  de error `unapproved_license` (un fixture sintético que deja de declarar la
  aprobación también se rechaza).
- Un manifest cuyos checksums no coinciden, cuyos IDs están duplicados o cuyo
  formato/versión no está soportado se rechaza con el código `invalid`
  correspondiente.
- El pipeline mapea estos fallos a razones de abstención (`blocked_corpus`,
  `invalid_corpus`, `stale_corpus`) y nunca sirve una respuesta de un corpus
  excluido por el gate.

## Gate obligatorio 2 — auditoría de source manifest (nuevo límite)

El límite de source manifest es una ruta de desarrollo local SEPARADA, explícita
y opt-in. Lo ejercitan solo `scripts/acquire_corpus.py` y
`scripts/verify_corpus.py`; no está conectado a `load_corpus`, al pipeline, a la
demo ni a la evaluación sintética. FALLA CERRADO:

- Un source manifest debe ser válido de esquema (todos los campos de evidencia
  presentes y bien tipados, al menos un archivo esperado, `document_count > 0`)
  o se rechaza (`malformed_source_manifest`,
  `unsupported_source_manifest_version`).
- Solo se aceptan `approved-for-local-development` o
  `approved-for-redistribution`; `blocked`, `under-review` y los estados
  desconocidos se rechazan (`blocked_corpus_status`).
- Solo se acepta el corpus comprometido en `APPROVED_CORPORA`
  (`python-docs-3.12.13`); cualquier otro id/versión de corpus se rechaza
  (`unapproved_source`).
- La revisión y el nombre Y versión de la licencia deben coincidir exactamente
  con la aprobación comprometida (`revision_mismatch`, `license_mismatch`); una
  revisión mutable es malformada.
- `redistribution_status` debe ser un valor conocido y consistente con el estado
  del corpus: `approved-for-local-development` no puede reclamar
  redistribución, y `approved-for-redistribution` debe usar el estado de
  redistribución coincidente y llevar una justificación explícita
  (`invalid_redistribution_status`, `missing_redistribution_justification`).
- Los archivos esperados deben existir con tamaños de bytes exactos y valores
  SHA-256 crudos (`source_file_missing`, `source_file_size_mismatch`,
  `source_file_checksum_mismatch`).
- El resultado del escaneo de privacidad debe coincidir con el `privacy_status`
  del manifest (`privacy_audit_failed`).

Los códigos estables están definidos en `src/evidence_grounded_rag/corpus/errors.py`
(la jerarquía `SourceManifestError`).

## Auditoría de privacidad (límite de source manifest)

`src/evidence_grounded_rag/corpus/privacy.py` escanea cada archivo capturado en
busca de marcadores de alta precisión de contenido privado o sensible (emails,
teléfonos, SSNs, patrones de tarjetas de crédito y pasaportes, palabras clave
financieras/médicas/credenciales/secretos). Los hallazgos nunca llevan el texto
coincidente. Cualquier documento con un hallazgo se **excluye** de los datos
derivados — la exclusión es a prueba de fallos. El resultado de la auditoría debe
coincidir con el `privacy_status` del manifest o la verificación falla.

Los archivos aprobados se escanearon en la adquisición (2026-08-11):
`no-private-content-detected`, cero documentos excluidos.

## Requerido por corpus (cualquier corpus real futuro)

| Propiedad | Requisito |
| --- | --- |
| Procedencia | URL(s) de fuente y fecha de recuperación registradas |
| Licencia | Términos documentados textualmente con cita |
| Derechos de reutilización | Explícito: indexación, transformación, redistribución |
| Atribución | Crédito requerido registrado |
| Límites de volumen | Restricciones de tamaño o tasa documentadas |
| Versionado | El snapshot del corpus es reproducible y versionado |

Un corpus futuro que no pueda fijarse a una única revisión exacta con una única
licencia estable debe rechazarse bajo el mismo razonamiento usado para Rust Book
y Kubernetes arriba.

## Excepción de fixture sintético (estado actual)

Pequeños fixtures escritos a mano para desarrollo offline están permitidos bajo
la licencia MIT de este repositorio, y son claramente sintéticos:

- Fixture comprometido: `data/fixtures/synthetic_manifest.json` (id de corpus
  `synthetic-fixtures`, versión `1.0.0`), 6 documentos ficticios / 11 fragmentos
  sobre la ficticia "Aurora City". Regenerado determinísticamente por
  `scripts/materialize_fixtures.py`.
- El manifest declara `provenance: project-generated-synthetic`,
  `synthetic: true`, `real_corpus: false`, estado de licencia `approved`
  (contenido generado por el proyecto cubierto por la licencia MIT del repo).
- Cada resultado, reporte y la demo se etiquetan **sintético/offline**.
- Las ubicaciones de fragmentos usan el esquema `synthetic://`; sin localizadores
  externos.

## Aceptación

- [x] Gate de licencia implementado, fail-closed y testeado (corpus real
      bloqueado, licencia no aprobada, desajuste de checksum, IDs duplicados,
      errores de formato).
- [x] Los fixtures son claramente sintéticos, pequeños y versionados; nunca
      representan un corpus real.
- [x] Un corpus local real aprobado (docs de Python 3.12.13), solo para
      desarrollo local, con candidatos/rechazos y evidencia de licencia
      registrados aquí.
- [x] Auditoría de source manifest implementada, fail-closed, testeada offline y
      mantenida separada del gate sintético y el pipeline de servicio.
- [x] Auditoría de privacidad implementada; los documentos excluidos no pueden
      entrar en los datos derivados.
- [x] Ningún corpus Gob.ec ni otro corpus sin licencia está presente en el
      repositorio.
- [ ] Términos de licencia documentados textualmente para cualquier corpus real
      candidato.
- [ ] Juicio de compatibilidad registrado con razonamiento.
