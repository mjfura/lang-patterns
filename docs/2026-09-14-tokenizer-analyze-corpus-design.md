# Análisis a priori de un corpus: tokenización y dominios

**Objetivo:** antes de entrenar (o de reentrenar con texto nuevo), saber cómo tokeniza un corpus y
si pertenece a un único dominio de conocimiento o a varios. `make tokenizer-analyze-corpus
MODEL=<nombre>` no toca el modelo, su tokenizer ni sus datos: aprende un tokenizer en memoria,
mide, agrupa y escribe un reporte.

Es un análisis, no un guardagujas todavía: usar el veredicto de dominio para decidir si un texto
emergente dispara el aprendizaje queda para una fase posterior.

## Configuración — `configs/models/<nombre>/tokenizer_analysis.json`

```json
{
  "files": ["data/knowledge_domains/kid_learning/geometric_shapes_train.txt"],
  "tokenizer": "reserved",
  "merge_budget": 1000,
  "unit": "document",
  "encoder": "sentence-transformers/all-MiniLM-L6-v2",
  "max_clusters": 5,
  "centroid_distance": 0.8,
  "seed": 0
}
```

`files` es obligatorio; el resto tiene esos valores por defecto. `tokenizer` es `reserved` o
`byte_level`; `unit` es `document` (lo que separa `<|endoftext|>`, o cada línea) u `sentence`.

## Tokenización — `src/eva_llm/analysis/tokenization.py`

El tokenizer se aprende con el mismo código que el pipeline (`ReservedBPETokenizer.trained` o
`ByteLevelBPETokenizer.from_document_sequence`, marcadores reconocidos) y se descarta al terminar.
Métricas sobre los tokens de contenido de cada documento (sin `<bos>`/`<eos>`):

- documentos, palabras, bytes UTF-8, tokens;
- merges aprendidos de `merge_budget`, `vocab_size`, tokens distintos usados y su fracción del
  vocabulario;
- tokens por palabra y bytes por token (compresión);
- entropía unigrama en bits por token, su valor normalizado (sobre log₂ de los tokens usados),
  perplejidad (2^H) y bits por byte;
- tokens vistos una sola vez.

`tokens.tsv` lista cada merge aprendido, en orden, y cada token usado: id, tipo (special, byte,
merge), texto, partes del merge, conteo y frecuencia.

## Dominios — `src/eva_llm/analysis/domains.py`

1. Unidades: documentos u oraciones. Antes de codificar se quitan los marcadores de turno
   (`<|im_start|>rol` y `<|im_end|>`): si no, el encoder agrupa por formato y no por tema.
2. Encoder estático (`sentence-transformers`, all-MiniLM-L6-v2 por defecto, embeddings
   normalizados). Se carga al usarlo; la primera vez descarga el modelo. Detrás de un Protocol, así
   los tests usan uno falso sin red.
3. K-means para k = 2…min(`max_clusters`, unidades ÷ 5): como máximo un grupo por cada 5 unidades,
   y con menos de 10 unidades el veredicto es indeterminado. Sin ese tope, los 12 documentos de
   figuras (un solo tema) se partían en 5 grupos, uno de un solo documento.
4. **Veredicto por distancia entre centroides:** k dominios para el mayor k cuyos pares de
   centroides están todos a al menos `centroid_distance` (0.8) en coseno; un dominio si ni los dos
   grupos de k = 2 llegan. El Silhouette se reporta pero no decide.
5. Reporte: Silhouette y distancia mínima entre centroides por k, veredicto, tamaño de cada grupo,
   distancias entre centroides, texto más cercano a cada centroide y la asignación de cada unidad.
   Un dominio hallado con menos de 5 unidades se marca como débil.

## Calibración (all-MiniLM-L6-v2, k = 2, 5 muestras por condición)

| corpus | Silhouette | distancia entre centroides |
|---|---|---|
| un dominio: figuras (14) | 0.312 | 0.344 |
| un dominio: preguntas Python (14 / 40) | ≤ 0.181 | ≤ 0.761 |
| un dominio: documentación Python (40) | ≤ 0.079 | ≤ 0.594 |
| un dominio: habla infantil (40) | ≤ 0.149 | ≤ 0.523 |
| mismo tema, dos registros: preguntas + documentación Python | ≤ 0.114 | ≤ 0.580 |
| dos dominios: figuras + Python | ≥ 0.340 | ≥ 0.816 |
| dos dominios: figuras + infantil | ≥ 0.343 | ≥ 0.861 |
| dos dominios: Python + infantil | ≥ 0.219 | ≥ 0.860 |
| dos dominios: documentación + infantil | ≥ 0.172 | ≥ 0.900 |
| intrusión: 14 figuras + 5 Python | ≥ 0.415 | ≥ 0.854 |

K-means con k = 2 separó exactamente los dos orígenes en todas las mezclas (ARI 1.00). El Silhouette
se superpone entre un dominio y dos (figuras sola 0.312 frente a mezclas desde 0.172): depende de lo
compacto de cada dominio. La distancia entre centroides no se superpone; 0.8 queda entre 0.761 y
0.816. Margen estrecho, medido en inglés con un solo encoder y solo para k = 2.

## Salida

Resumen en terminal; `analysis/<nombre>/<fecha-hora>/` con `report.md`, `tokens.tsv` y
`domains.json`. `analysis/` está en `.gitignore`.

## Dependencias

`sentence-transformers` y `scikit-learn`. Sin tipos: mypy los ignora con un override. Los tests no
usan red.

## Relación con `analyze-corpus`

Se mantiene aparte: sigue siendo la estimación rápida de `merge_budget` por conteo de palabras.
