# ¿La culpa es del corpus? Diagnóstico de la suficiencia sintáctica local de un corpus de entrenamiento antes de entrenar

*Borrador v0.2, 2026-10-15. Versión en español para revisión. La versión a publicar es
`draft-en.md`. Los números salen de los registros de experimentos en
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md`, `docs/superpowers/specs/2026-10-11-settings-from-data-design.md`
y las carpetas `experiments/` de este repositorio. Cada experimento se prerregistró en esos
documentos antes de correrlo, y las predicciones que fallaron se reportan como fallidas. La v0.2
reemplaza los umbrales del indicador fijados a mano por ajustes derivados de los datos, agrega una
lectura sin referencia y vuelve a correr los experimentos del lado del corpus con la nueva
herramienta. Los números del lado del modelo no cambian respecto de la v0.1.*

---

## Título (propuestas)

1. **¿La culpa es del corpus? Diagnóstico de la suficiencia sintáctica local de un corpus de entrenamiento antes de entrenar** (preferido)
2. Servido o ausente: un diagnóstico previo al entrenamiento de los patrones sintácticos en corpus para modelos de lenguaje
3. Marca el corpus, no el modelo: atribuir las fallas de generalización sintáctica a los datos de entrenamiento

*(En inglés: "Is the Corpus to Blame? Diagnosing the Local Syntactic Sufficiency of a Training
Corpus Before Training".)*

---

## Resumen

Cuando un modelo de lenguaje no generaliza un patrón de su dominio, la causa puede estar en el
modelo (su arquitectura, tamaño o entrenamiento) o en el corpus (el patrón nunca se mostró con
suficiente variedad para aprenderse). Las medidas existentes de calidad de datos no distinguen
entre ambas. Dan un solo puntaje por documento o por dataset, y la más común, la perplejidad en
texto no visto, puede ordenar los corpus al revés de cómo generaliza el modelo entrenado con
ellos. Proponemos un **indicador de suficiencia sintáctica local, calculado solo sobre el
corpus**. Lee cada patrón sintáctico del corpus en dos niveles: una cabeza seguida de una clase
gramatical (*she* → verbo), y una clase de cabezas seguida de sus rellenos (pronombre → verbo),
con clases inducidas del corpus. Dado un corpus de referencia del dominio, cada patrón se compara
con muestras aleatorias de la referencia del mismo tamaño y se reporta como insuficiente
(*under-shown*), ausente o dentro del ruido. Sin referencia, cada cabeza se compara con sus
hermanos en el propio árbol de cabezas del corpus, y los patrones que el corpus muestra
débilmente se reportan como débiles (*weak*). **Ningún umbral se fija a mano.** La tokenización,
el tronco bajo el cual un patrón no se lee y el número de muestras aleatorias salen de los datos,
y el único ajuste que queda es el nivel de la prueba (alpha = 0.06 por familia de veredictos).

Validamos el indicador con intervenciones sobre el corpus y modelos entrenados, con predicciones
escritas antes de cada experimento:

- En una gramática sintética de recetas, el indicador predice qué patrones verbo–sustantivo
  generaliza un modelo (Spearman 0.94–0.98), mientras que la pérdida en texto no visto ordena los
  corpus al revés.
- En texto natural (FineWeb), la lectura por cabeza encuentra huecos plantados sin falsas alarmas
  en texto sin plantar. En las cabezas plantadas, un modelo pierde 1–4.6 nats en la palabra
  siguiente verdadera.
- La sintaxis se aprende a nivel de clase. Quitarle un patrón a dos cabezas les cuesta entre el
  25 y el 57% de lo que les cuesta quitárselo a toda su clase. Quitárselo a la clase hace perder la
  sintaxis a todos sus miembros (−0.13 a −0.30 en un puntaje de pares mínimos), en pronombres,
  modales y posesivos.
- **Atribución.** Cuando el indicador marca un patrón de clase, ninguna configuración de modelo
  que probamos lo recupera, y los modelos más grandes o entrenados más tiempo lo hacen peor.
  Cuando no marca nada, un corpus del mismo tamaño alcanza la sintaxis de la referencia con una
  diferencia de 0.02. El resultado de los modales se mantiene, y crece, con diez veces más datos
  (≈10M de palabras, BLiMP 0.63).
- **Ajustes.** Los veredictos no cambian con la granularidad entre 1.35 y 1.80 tokens por
  palabra, con 250 a 4,000 tokens de contexto, ni con 8 a 64 clases de relleno. Los cortes
  aleatorios de la referencia se marcan a la tasa de diseño (2 de 48).

Frente a métricas de un solo número (perplejidad, KL de unigramas, MATTR, distinct-2, puntaje
Vendi), el indicador localiza el patrón dañado. Además detecta pérdidas específicas de 2,000–3,600
oraciones que dejan a esas métricas dentro de su ruido. Su alcance es la sintaxis local (de
adyacencia), validada con modelos de hasta 6.2M de parámetros.

---

## 1. Introducción

Quien entrena un modelo de lenguaje con un corpus de dominio y ve que no generaliza enfrenta una
pregunta de diagnóstico antes que cualquier pregunta de ingeniería. ¿Al corpus le falta algo, o
el modelo no es el adecuado? Las dos respuestas piden acciones opuestas. Recolectar o rebalancear
datos no arregla un modelo demasiado chico o mal entrenado. Ajustar la arquitectura no arregla un
corpus que nunca muestra un patrón con suficiente variedad. Hoy esa pregunta se responde, cuando
se responde, entrenando e inspeccionando.

Planteamos la pregunta a través de la noción de **dominio de conocimiento**: un subconjunto acotado
de los patrones sintácticos y semánticos de un idioma, como las recetas, que son sobre todo
acciones imperativas sobre ingredientes. Este artículo trata solo su capa sintáctica. La sintaxis
es aquí el marco bien formado de una oración, sea verdadera o no. "Corta la leche" es
sintácticamente correcta; que no tenga sentido es semántica, fuera de alcance. La premisa sigue
una observación de larga data en lingüística de corpus (las medidas de productividad de Baayen) y
en el aprendizaje de modelos de lenguaje (Wei et al., 2021; Misra & Mahowald, 2024). Un patrón se
aprende como patrón, y no como palabra memorizada, cuando el corpus muestra su posición con
suficientes rellenos distintos. Un corpus es **sintácticamente suficiente** para su dominio
cuando muestra así cada patrón local del dominio.

Las herramientas existentes no miden esto.

- **Los métodos de calidad y selección de datos** (filtrado por perplejidad al estilo CCNet,
  DSIR, QuRating, el coeficiente de diversidad) dan un número por documento o dataset, pensado
  para perplejidad o desempeño en tareas. Un solo puntaje no puede decir qué patrón falta.
- **La pérdida en texto no visto puede engañar.** En una gramática controlada mostramos que ordena
  los corpus al revés de la generalización sintáctica: un corpus restringido también lo está en su
  partición de validación, así que parece más predecible.
- **El entrenamiento con corpus filtrados** (FiCT; Patil et al., 2024) y estudios afines quitan una
  construcción y miden qué aprende el modelo entrenado. Son posteriores al entrenamiento por
  diseño, y su mensaje principal, que los modelos generalizan desde evidencia indirecta, hace que
  la pregunta de qué eliminaciones importan sea real.

Un diagnóstico de este tipo es tan confiable como sus ajustes. Un umbral ajustado sobre los corpus
en que se desarrolló el método puede producir, en un corpus nuevo, marcas que son artefactos del
umbral. Por eso exigimos que cada ajuste del indicador se derive de los datos o se demuestre que
no cambia sus veredictos, y reportamos las mediciones (§5.8).

Contribuimos:

1. **Un indicador**, calculado solo sobre el corpus. Lee patrones sintácticos locales a nivel de
   cabeza y en cada nodo de un árbol inducido de cabezas. Con referencia, los compara con muestras
   aleatorias de la referencia del mismo tamaño; sin ella, con las cabezas hermanas del corpus
   (§3). Viene empaquetado como herramienta de línea de comandos y como interfaz de usuario.
2. **Ajustes a partir de los datos.** Un rango medido de granularidad del tokenizador, un nivel de
   prueba que fija el número de muestras aleatorias y un tronco de lectura elegido por el criterio
   de Tarone reemplazan todo umbral fijado a mano (§3.4, §5.8).
3. **Una validación causal** con intervenciones sobre el corpus, en una gramática sintética y en
   texto natural: huecos plantados, dosis–respuesta, filtros realistas, y eliminación de un patrón
   a una cabeza frente a toda su clase (§5).
4. **Un hallazgo sobre el nivel en que se aprende la sintaxis.** Una cabeza que pierde su posición
   conserva la sintaxis mientras su clase conserve el patrón. Una clase que pierde el patrón pierde
   la sintaxis de todos sus miembros (§5.4).
5. **Una prueba de atribución.** Un patrón de clase marcado es una falla del corpus que ninguna
   configuración de modelo probada arregla. Un corpus sin marcas permite alcanzar la sintaxis de
   la referencia (§5.5). Esta es la garantía que necesita quien trabaja con estos modelos.
6. **Comparaciones con métricas de corpus de un solo número y con BLiMP**, con un recuento honesto
   de dónde ven lo mismo y dónde no (§5.7), y una lista de todas las predicciones fallidas
   (Apéndice A).

---

## 2. Trabajo relacionado

**Calidad y selección de datos para preentrenamiento.**

- El filtrado por perplejidad contra un modelo de referencia (el pipeline CCNet) es un filtro de
  calidad estándar.
- DSIR (Xie et al., 2023) selecciona datos por remuestreo de importancia sobre n-gramas con hash.
- QuRating (Wettig et al., 2024) usa juicios de un LLM sobre calidad de escritura, experticia y
  valor educativo.
- El coeficiente de diversidad (Miranda et al., 2023) mide la variabilidad de un dataset con
  embeddings Task2Vec.
- Predictores de influencia sin entrenamiento, como DataProphet (2026), ordenan datasets de
  supervisión antes de entrenar.
- Albalak et al. (2024) revisan el área.

Todos califican documentos o datasets como un todo. Ninguno reporta qué patrones sintácticos sirve
mal un corpus, y ninguno separa una falla del corpus de una del modelo. Encontramos que los filtros
basados en perplejidad son los únicos recortes de un corpus que el indicador marca (§5.3).

**Aprender sintaxis de corpus manipulados.**

- Wei et al. (2021) muestran que BERT aplica la concordancia sujeto–verbo a pares nunca vistos
  juntos, pero tiene problemas con ítems poco frecuentes y con sesgos fuertes.
- FiCT (Patil et al., 2024) quita una construcción del corpus de entrenamiento y encuentra que
  tanto LSTMs como transformers la generalizan desde evidencia indirecta. Luo & Steinert-Threlkeld
  (2026) lo extienden a la coordinación de categorías distintas.
- Misra & Mahowald (2024) muestran que una construcción rara (AANN, "a beautiful five days") se
  aprende de construcciones relacionadas más frecuentes, y mejor con más variabilidad en el input.

Nuestros resultados coinciden a nivel de cabeza: un pronombre que pierde su posición de verbo
conserva la sintaxis gracias a sus hermanos. Agregamos el nivel de clase, donde la eliminación sí
impide el aprendizaje, y un indicador que dice antes de entrenar qué eliminaciones importan.

**Medidas de productividad y diversidad.** Las medidas de productividad de Baayen usan los hápax
para estimar cuán abierta está una categoría morfológica. La masa faltante de Good–Turing
(*missing mass*: singletons sobre tronco) aplica la misma idea a una posición sintáctica;
mostramos que por sí sola no marca una posición empobrecida (§5.9). Las medidas de diversidad
léxica, sintáctica y semántica de las salidas de un modelo (Guo et al., 2024) evalúan texto
generado, no corpus de entrenamiento, y no están ligadas a lo que el modelo aprende.

**Pruebas múltiples con estadísticos discretos.** El indicador prueba decenas de miles de patrones
a la vez. Su umbral es el estadístico máximo sobre muestras aleatorias (un umbral de error por
familia basado en remuestreo), y los patrones demasiado raros para marcarse alguna vez quedan fuera
de la familia, siguiendo a Tarone (1990).

**Evaluación sintáctica dirigida.** Los pares mínimos (Linzen et al., 2016; Marvin & Linzen, 2018),
la evaluación sistemática de Hu et al. (2020) y BLiMP (Warstadt et al., 2020) evalúan modelos ya
entrenados. Nuestra lectura del lado del modelo es un puntaje de pares mínimos construido para
excluir la semántica. Rellenos de la clase correcta nunca vistos en esa posición compiten contra
palabras agramaticales de frecuencia equivalente, con clases de un etiquetador externo (§4.3).
También reportamos BLiMP.

---

## 3. El indicador

### 3.1 Unidades: tokens, transiciones, clases

- **Tokens.** El corpus se tokeniza con un BPE a nivel de byte aprendido sobre el texto del
  dominio (el de la referencia o, sin ella, el del corpus), con las menos fusiones que dejan 1.50
  tokens por palabra. Se midió que los veredictos no cambian entre 1.35 y 1.80 (§5.8). En el
  conjunto de FineWeb esto son 11,381 fusiones.
- **Todo token participa.** Todo token que aparece en al menos dos documentos puede ser cabeza o
  relleno: palabras, partes de palabras, puntuación y números.
- **Transiciones sintácticas y léxicas.** Una transición es el último token de la cabeza seguido
  del relleno. Es *sintáctica* cuando el relleno empieza un nuevo pre-token del tokenizador (una
  palabra nueva, una secuencia de puntuación, un número o una contracción), y *léxica* cuando
  continúa la palabra del token anterior (` model` → `s`). Ninguna fusión cruza un límite de
  pre-token, así que la regla es exacta. Los dos tipos se cuentan como patrones separados, y el
  diagnóstico son las filas sintácticas.
- **El árbol de clases.** Cada token se describe por la distribución de sus vecinos a la izquierda
  y a la derecha sobre los 1,000 tokens más frecuentes. Las filas se toman en raíz cuadrada, para
  que la distancia euclídea sea la distancia de Hellinger, y se agrupan con el método de Ward en
  un solo árbol.
- **Clases de relleno.** El árbol se corta en el mayor salto de altura entre 8 y 64 clases (12 en
  FineWeb). En la gramática de recetas ese corte recupera exactamente las categorías de la
  gramática. Un corte más fino empieza a agrupar los rellenos por significado (lo que se vierte
  frente a lo que se pica), lo que metería semántica en una medida sintáctica.
- **Clases de cabezas.** Cada nodo interno del mismo árbol es una clase de cabezas. No se elige
  ningún corte: los patrones de clase se leen en cada nodo (§3.2).

### 3.2 Patrones

- **Patrón de cabeza.** Una cabeza de uno o dos tokens seguida de un relleno de la clase Y, con su
  tipo de transición.
- **Las apariciones se cuentan en contextos distintos** (el token anterior a la cabeza y el
  posterior al relleno), así que las copias exactas no suman.
- **Tronco.** El número de contextos distintos de un patrón.
- **Patrón de clase.** Un nodo del árbol de cabezas, con los rellenos que siguen a cualquiera de
  sus cabezas de un token, juntos.

### 3.3 Lecturas

- **Diversidad:** rellenos / rellenos esperados, donde rellenos esperados = Σ_y [1 − (1 − p_y)^n],
  con n el tronco y p_y la proporción de cada miembro de la clase en el corpus. Compara cabezas de
  distinta frecuencia contra extracciones libres de la clase.
- **Divergencia D:** KL(R ‖ S), en nats. R es la distribución de rellenos de la posición en la
  referencia. S es la del corpus, suavizada con Witten–Bell hacia el unigrama de rellenos del
  corpus, sumando un conteo por cada relleno de la referencia. D tiene las unidades de la
  pérdida: un modelo que aprendió S y se evalúa en texto de R pierde en promedio KL(R ‖ S) nats por
  aparición de la posición. En los huecos plantados, la caída en la log-probabilidad del modelo es
  −0.68 · D (R² = 0.77).
- **Recortes a igual tamaño.** Toda lectura depende del tamaño del corpus, así que un corpus más
  chico siempre sale peor. Por eso cada corpus se compara con muestras aleatorias de documentos
  enteros de la referencia, del mismo tamaño en bytes. La referencia se tokeniza y clasifica una
  sola vez y se reutiliza para cada corpus que se diagnostica contra ella.

### 3.4 Veredictos con referencia

El t de cada patrón es cuánto se aleja el corpus de la media de los recortes, en la dispersión de
los recortes: la diversidad en las filas de cabeza, el tronco en `absent` y D en las filas de
clase. La dispersión incluye el redondeo de la propia lectura (la diversidad se mueve de a un
relleno, un tronco de a uno): sd' = √(sd² + step²/12).

**El umbral sale de los recortes.** Cada recorte, por turno, se deja afuera y se lee contra los
demás, y se registra su mayor t sobre todos los patrones. Un corpus se marca solo donde se aleja
más que el k-ésimo mayor de esos máximos: una muestra aleatoria de la referencia pasa entonces el
umbral con probabilidad de alrededor de k / (n + 1).

- **alpha**, la probabilidad de que una muestra aleatoria de la referencia se marque en una
  familia de veredictos, es el único ajuste que declara el usuario. Por defecto es 0.06 por
  familia (filas de cabeza, `absent`, filas de clase). Es el nivel de la prueba, una tolerancia y
  no una propiedad de los datos.
- **El número de recortes** se deduce de alpha. El k-ésimo mayor de n máximos de los recortes
  también tiene ruido: entre conjuntos de recortes, la probabilidad de pasarlo sigue una
  Beta(k, n + 1 − k). La herramienta usa la menor cantidad de recortes que mantiene esa
  probabilidad entre alpha/2 y 1.5 · alpha en el 80% de los conjuntos de recortes: 101 recortes y
  k = 6 con alpha 0.06.
- **El tronco de lectura** sale de los recortes, por el criterio de Tarone. Un patrón demasiado
  raro para marcarse aun si el corpus mostrara un solo relleno no aporta ningún veredicto, y su
  ruido sube el umbral de todos los demás patrones. El tronco de lectura mínimo es el que deja más
  patrones que podrían marcarse.

**Filas de cabeza:**

- **insuficiente (*under-shown*):** diversidad bajo la media de los recortes por más que el umbral;
- **ausente (*absent*):** el tronco bajo el tronco medio de los recortes por más que el umbral de
  `absent`. Una muestra aleatoria de este tamaño mostraría el patrón, y el corpus muestra
  demasiado poco de él;
- **dentro del ruido (*within-noise*)** en los demás casos, y **sin lectura (*unread*)** bajo el
  tronco de lectura.

**Filas de clase:** D sobre los rellenos juntos del nodo, frente a los recortes. Un nodo es
**insuficiente** cuando su t supera el umbral de clase. De cada rama se reporta el nodo con el t
más alto, así que el reporte nombra el grupo de cabezas más chico que carga el desplazamiento.
Cada fila de clase lista además sus **cabezas en falta**: los miembros cuyas propias filas son
insuficientes o ausentes.

### 3.5 Veredictos sin referencia

Un patrón que el corpus muestra débilmente es débil para un modelo entrenado con él, haga lo que
haga el dominio. Un corpus solo puede mostrarlo comparando una cabeza con sus hermanos:

- Cada hijo de un nodo del árbol de cabezas (una cabeza o una clase de cabezas) se compara con su
  padre, por clase de relleno y tipo de transición.
- **La hipótesis nula.** Si las apariciones de esa clase en el padre se hubieran repartido al azar
  entre sus dos hijos, ¿cuántos rellenos distintos mostraría este hijo? Se sortean 101 particiones
  aleatorias de forma exacta, sin reemplazo, con el tronco del hijo.
- El umbral se construye a partir de las particiones como en §3.4, con el mismo alpha, y solo se
  leen las pruebas que un solo relleno podría hacer fallar.
- Solo se prueban los hijos de un nodo que está dentro de una clase de relleno. Por encima del
  corte de clases, los dos lados de una partición son categorías distintas (`,`/`the`/`to` frente
  a `.`/`:`), y toda prueba así salía débil.
- Las cabezas de dos tokens se ubican por su primer token.

Veredictos: **débil (*weak*)**, más allá del umbral; **raro (*rare*)**, mostrado menos veces que
cualquier posición que la prueba pueda leer; **dentro del ruido**; **sin lectura**, sin hermano
contra el cual leerlo.

Tres casos son invisibles sin referencia. Un patrón que al corpus le falta por completo, y una
clase mostrada débilmente en su conjunto, no dejan ningún hermano más fuerte con el cual comparar.
Una cabeza tan restringida que el árbol la separa de sus hermanos pierde a su hermano. El reporte
declara estos límites y recomienda una referencia del mismo dominio. Una referencia además separa
la debilidad propia del corpus de la del dominio: la mayoría de los patrones débiles de un corpus
natural son su gramática (` to` frente a las otras preposiciones, verbos de reporte), y una
referencia los lee dentro del ruido.

### 3.6 La herramienta

`corpus-diagnose` (Python) recibe un corpus y, opcionalmente, una referencia. Escribe reportes en
JSON y Markdown: resumen, advertencias, clases, filas de cabeza, filas de clase y configuración,
incluidos alpha y el número de recortes.

- Sin presupuesto de fusiones, aprende el tokenizador del dominio hasta 1.50 tokens por palabra,
  retomando el BPE en rondas que se duplican.
- Un `--merges N` dado se revisa antes del diagnóstico: fuera de 1.35–1.80 tokens por palabra, el
  comando se niega y nombra la granularidad alcanzada, salvo que se pase `--force`.

Una página de Streamlit corre el mismo diagnóstico sobre cualquier dataset del catálogo.

---

## 4. Diseño experimental

### 4.1 Corpus

- **Gramática de recetas.** Una gramática probabilística con léxico cerrado (verbos imperativos,
  ingredientes contables, adjetivos, adverbios, utensilios) y cuatro reglas de oración. Una
  perilla cambia con cuántos ingredientes aparece cada verbo (1, 2, 4, 8, 16, 20, 24 o todos). Cada
  ingrediente sigue apareciendo en otras oraciones. Las combinaciones verbo–ingrediente reservadas
  nunca se ven y sirven como ítems de prueba.
- **FineWeb** (Penedo et al., 2024).
  - El conjunto principal F son 4,600 documentos (14.4 MB, ≈2.4M de palabras, 3.4M de tokens) de
    los primeros 5,000 descargados en streaming. Los otros 400 se reservan para evaluación.
  - Una muestra de 6.5 MB en 8 partes se usa para los plantados.
  - BigF son los primeros 20,000 documentos menos los de evaluación (57.8 MB, ≈10M de palabras).
- **Intervenciones.**
  - Plantados: los seguidores de ciertas cabezas se reemplazan por uno de sus k rellenos más
    frecuentes, o se quitan las oraciones en que toman otro relleno.
  - Recortes de documentos del 10–50% de F: al azar, filtros de perplejidad, filtros de largo.
  - Recortes a nivel de oración sobre un recorte del 50% de F (r50), que quitan toda oración en
    que una cabeza de la lista va seguida inmediatamente de cierta clase según spaCy. Se hacen a
    nivel de clase, a nivel de cabeza (solo dos miembros) y como controles aleatorios, todos con
    el mismo tamaño en bytes.
  - **Cortes etiquetados** para la herramienta: los cortes de clase y de cabeza de pronombres y
    modales, posesivos, números y verbos base (8 cortes), 12 controles aleatorios, y dos cortes
    que borran cuatro palabras cada uno.

### 4.2 Modelos

Un decodificador con la arquitectura de GPT-2 y cabeza atada, entrenado en CPU.

- **pequeño:** 2 capas, 128 de ancho, 4 cabezas, contexto 128, 1.9M de parámetros, 3,000 pasos de
  32 × 128 tokens, tasa de aprendizaje 1e-3.
- **mediano:** 4 capas, 256 de ancho, 6.2M de parámetros, tasa de aprendizaje 5e-4.
- **largo:** el pequeño, con 9,000 pasos.
- **escala:** el mediano, 30,000 pasos sobre BigF, unas 5 horas por corrida.
- Cada corpus aprende su propio BPE (11,563 fusiones en FineWeb). El mejor checkpoint se elige con
  una fracción de validación. Semillas 0 y 1, salvo que se indique otra cosa.

### 4.3 Lecturas del lado del modelo

- **Caída (*drop*):** para una cabeza, la media sobre sus apariciones de evaluación de
  log p_modelo(palabra siguiente verdadera) − log p_modelo de referencia, en nats. Es una lectura a
  nivel de relleno: mezcla sintaxis con semántica y tema.
- **Razón de generalización (gramática de recetas):** después de "⟨bos⟩ cut the", la probabilidad
  que se da a los ingredientes nunca vistos con ese verbo, relativa a su proporción (1 significa
  que el patrón se aprendió, 0 que se memorizó).
- **G (texto natural):** después de un prefijo real de evaluación que termina en la cabeza h, la
  proporción de comparaciones en que el modelo prefiere una palabra de la clase de la posición de
  h que nunca sigue a h en la referencia, frente a una palabra de una clase agramatical de
  frecuencia equivalente. La verdad no cuenta: después de "she", cualquier verbo en pasado es
  válido. Las clases vienen de spaCy (`en_core_web_sm`), nunca del indicador. Contrastes
  gramaticales:
  - pronombres → verbos en pasado (VBD), contra sustantivos y adjetivos;
  - modales → verbos en forma base (VB), contra sustantivos y adjetivos;
  - posesivos → sustantivos, contra adverbios y verbos VBD/VBZ.
- **BLiMP** (67 paradigmas × 1,000 pares), por log-probabilidad total.

### 4.4 Protocolo

El diseño, las predicciones y los umbrales de cada experimento se escribieron en los documentos de
diseño antes de correrlo. Los parámetros de construcción solo se cambiaron a partir de conteos del
corpus, antes de puntuar cualquier modelo, y cada cambio quedó registrado. Las predicciones
fallidas se conservan y se listan en el Apéndice A.

El indicador se construyó primero con umbrales fijados a mano (un piso de diversidad de 0.5, un
margen de 3 sigma sobre 8 recortes, un tronco mínimo de 20, un piso de clase de 0.5 nats). Después
se reemplazaron por los ajustes de §3.4–3.5, y cada experimento del lado del corpus se volvió a
correr con la herramienta final. Los modelos no se reentrenaron: los corpus son los mismos, y solo
cambió su lectura. Donde un número del lado del corpus en §5 todavía viene de la primera versión,
está marcado.

---

## 5. Resultados

### 5.1 Las medidas en texto no visto no ven la insuficiencia sintáctica (gramática de recetas)

| Medida | Resultado |
|---|---|
| Cobertura por verbo vs. razón de generalización | Spearman 0.891 (una semilla); 0.940–0.975 con 3 semillas × 2 tamaños |
| Razón media, todos los ingredientes vs. uno por verbo | 0.909 vs. 0.011 |
| Pérdida en texto no visto y predictibilidad n-grama | Ordenan las variantes **al revés** en toda semilla y tamaño |

El modelo generaliza tarde:

- La razón del modelo pequeño con 16 rellenos es solo 0.17–0.19, y cruza 0.5 con una cobertura de
  0.67–0.80.
- El modelo más grande necesita más: 0.554 contra 0.901 con 24 rellenos. "Suficiente" depende del
  tamaño del modelo, punto al que volvemos en §6.
- Los pares mínimos del tipo "cut the tomato / cut the slowly", pensados como control fácil,
  siguieron a la razón (0.25 con un relleno) en vez de quedarse cerca del 100%.

### 5.2 Nivel de cabeza en texto natural: huecos plantados y dosis

**Detección, con la herramienta final.** Se plantaron cinco cabezas de una muestra de FineWeb
(*people, she, But, only, even*) en una parte, y cada parte se diagnosticó contra la muestra.

| Plantado | Cabezas encontradas (de 5) | Otras filas marcadas |
|---|---|---|
| oraciones quitadas, k = 2 | 4 | 0 |
| rellenos reemplazados, k = 2, 4, 8, 16 | 5 en cada k | 0 |
| rellenos reemplazados, k = 32 | 4 | 0 |
| 7 partes sin plantar | 0 | 0 |

- Toda cabeza encontrada se encuentra por su fila de rellenos determinantes, con diversidad
  0.03–0.17 para k = 2 y 0.27–0.35 para k = 32, frente a alrededor de 0.9 en las partes sin
  plantar. La mayoría también se lee `absent` en una segunda clase de relleno.
- Ninguna fila fuera de las cabezas plantadas se marca en ninguna parte, de 2,900–6,600 filas
  leídas.
- El nivel de clase marca una fila de clase en cada una de dos de las siete partes sin plantar
  (§7).
- Con el margen de 3 sigma de la primera versión, el mismo barrido encontró 5, 5, 5, 5 y 4 cabezas
  en FineWeb, y 19, 20, 19, 17 y 12 de 20 cabezas en cuatro fuentes (FineWeb, Wikipedia, arXiv,
  C4). Quitando oraciones encontró 3 de 5: dos cabezas que quedaron con 6 y 23 apariciones se
  leyeron apenas por encima de un piso fijo de 0.5.

**Costo en el modelo.**

- Los modelos entrenados con los corpus plantados dan a la palabra siguiente verdadera 2.17–4.60
  nats menos (k = 2) y 1.12–3.60 nats menos (k = 16), en toda cabeza plantada y semilla.
- Las cabezas de comparación quedan dentro de 0.3 nats en 19 de 20 lecturas.
- La pérdida sobre todo el texto se mueve solo 0.03 bits por byte: registra el daño, pero no dice
  dónde.

**Dosis.** Se plantaron catorce cabezas con diversidad 0.1–0.7, en dos asignaciones y dos
semillas.

- La pérdida es monótona en la dosis (Spearman 0.76–0.92).
- Las cabezas en 0.1 pierden 2.8–3.3 nats.
- El extremo suave no sale gratis: 0.38–0.57 nats en 0.5–0.7. Por lo tanto, la diversidad no es el
  tamaño del efecto.
- D sí lo es (caída ≈ −0.68 · D, R² 0.77, lectura de la primera versión), aunque en los
  plantados no ordena las cabezas mejor que la diversidad.

### 5.3 Recortes realistas de un corpus

Cinco recortes del 25–50% de F se diagnosticaron contra F, y se entrenaron seis modelos por
semilla (§4.2).

| Recorte | Tokens/palabra | Insuficientes | Ausentes | Filas de clase marcadas |
|---|---|---|---|---|
| aleatorio 50% | 1.49 | 0 | 0 | 0 |
| aleatorio 25% | 1.49 | 0 | 0 | 0 |
| aleatorio 10% | 1.52 | 0 | 0 | 0 |
| 25% menos parecido a Wikipedia | 1.39 | 2 | 11 | 4 |
| 25% más formulaico (menor perplejidad) | 1.61 | 1 | 4 | 1 |

**Los recortes aleatorios salen aleatorios.** No marcan nada, ni filas ni clases. El margen de
3 sigma de la primera versión marcaba 2–8 filas en cada uno, a la tasa que le daba el azar.

**Los filtros son los únicos recortes marcados**, pero levemente. El filtro tipo Wikipedia marca
"," y ":" antes de determinantes y etiquetas; el filtro formulaico marca ". Our". Sus filas
ausentes son unas pocas cabezas pobres ("50", " a day", " they should"). Habíamos predicho al
menos 10 filas insuficientes para cada filtro, y ambos fallaron.

**El z de una cabeza predice lo que el recorte le cuesta al modelo mejor que su frecuencia.** En el
recorte del 10%, Spearman(z, caída) sobre 416 cabezas es 0.157 / 0.181 (dos semillas), y el log
del tronco solo, 0.037 / 0.018. Con el margen de 3 sigma era 0.10 / 0.12, con el tronco en
0.10 / 0.11.

**Los filtros mueven las posiciones.** El puntaje de un recorte es el exceso de sus cabezas,
E = D − media de D en 8 recortes aleatorios de F de su tamaño, ponderado por las proporciones de
rellenos de F; z lo divide por la dispersión de los recortes dejando uno fuera. El filtro de
perplejidad que conserva el 25% de documentos más formulaicos da z = +18.6. El que conserva los
documentos menos parecidos a Wikipedia da z = +24.8. El filtro tipo CCNet con Wikipedia da
z = +9.7. Los recortes aleatorios dan |z| ≤ 1.5. Con la lectura de tokens de tipo palabra de la
primera versión, estos valores eran +15.0, +20.6, +3.9 y |z| < 0.6.

**El modelo paga donde apunta el exceso.**

- Dentro de un recorte filtrado, las cabezas con mayor E pierden más de lo que cuesta un recorte
  aleatorio del mismo tamaño: Spearman(E, caída − caída aleatoria) de −0.24 a −0.37, en seis
  filtros y ambas semillas (de −0.21 a −0.38 en la primera versión).
- El aparente éxito de D crudo en recortes aleatorios era solo tamaño de muestra. En el recorte del
  10%, D da −0.42 / −0.45 frente a la caída, y E solo −0.05 / −0.11.

**Elegir entre recortes.**

- Sobre diez recortes del 25%, el puntaje del corpus ordena solo en parte los bits por byte de los
  modelos: Spearman 0.65 / 0.55, frente a 0.5 predicho en ambas semillas (0.55 / 0.45 en la
  primera versión, que falló).
- El número de documentos, un indicador indirecto de la cobertura temática, pesa lo mismo
  (−0.47 / −0.44). El puntaje ordena lo que deja el número de documentos con 0.58 / 0.56.
- A lo largo de la dosis de un mismo filtro, con los documentos fijos, el puntaje sí ordena los
  modelos (Spearman 0.90). Cuando la dosis también suma documentos, no (0.60).

### 5.4 La sintaxis se aprende a nivel de clase

**Primera lectura, negativa.** Entre recortes realistas, G siguió al tamaño (F 0.748, recortes del
25% 0.700–0.736, del 10% 0.69) y al número de documentos, no al puntaje (−0.16 / −0.33). G por
cabeza se reproduce entre semillas (Pearson 0.87–0.91). La razón, que encontramos después: los
filtros realistas empobrecen la posición de una cabeza, pero nunca vacían el patrón de su clase.

**Eliminar en una cabeza frente a eliminar en la clase.** A partir de r50 y con el mismo tamaño en
bytes, se construyen tres corpus:

- `class_cut` quita toda oración en que un pronombre (*I you he she we they*) o un modal va
  seguido de un verbo;
- `head_cut` quita solo las de *she, they, can, will*;
- `control` quita oraciones al azar.

| Clase | ΔG corte de clase | Cabezas quitadas: corte de cabeza / corte de clase | Hermanos en el corte de cabeza |
|---|---|---|---|
| pronombres | −0.195 | −0.044 / −0.177 (25%) | −0.028 |
| modales (ítems VB) | −0.126 | −0.040 / −0.081 (49%) | −0.008 |
| posesivos | −0.303 | −0.194 / −0.343 (57%) | +0.046 |

- **Quitar el patrón de la clase hace perder la sintaxis a todos sus miembros.** Quitárselo a dos
  cabezas les cuesta entre un cuarto y algo más de la mitad, y deja intactos a sus hermanos.
- **El nivel de clase del indicador coincide con los modelos.**
  - En `class_cut` la herramienta reporta el nodo de los modales (t 319) y el nodo de pronombres
    *I you we they* (t 254), frente a un umbral de clase de 13.9.
  - En `head_cut` no reporta ningún nodo de pronombres, y sí un nodo de cinco modales que contiene
    *can* y *will* con t 33, un décimo del del corte de clase.
  - D por cabeza, en cambio, dice que *she* y *they* perdieron lo mismo en ambos cortes
    (+1.46 / +1.58).
- **La sintaxis se aprende en ambos niveles, sobre todo en el de clase.** Cuánto suma la evidencia
  propia de una cabeza depende de la clase: lo mínimo en pronombres, lo máximo en posesivos.

### 5.5 Atribución: ¿corpus o modelo?

Los cortes de clase marcados y sus controles sin marca del mismo tamaño se entrenan en tres
configuraciones (pequeño, mediano, largo), junto con F. La herramienta marca las filas de clase de
cada corte de clase (pronombres y modales t 254 y 319, posesivos t 165), y ninguna fila de clase de
ningún control.

| Clase | Mejor en F | Mejor en el control (sin marca) | Mejor en el corte marcado |
|---|---|---|---|
| pronombres | 0.981 | 0.961 | 0.846 |
| modales | 0.999 | 0.988 | 0.870 |
| posesivos | 0.872 | 0.859 | 0.600 |

- **Con marca.** Ninguna configuración recupera la clase (0.13–0.27 por debajo de F). Más
  capacidad lo empeora: pronombres 0.85 → 0.75 → 0.69, modales 0.87 → 0.80 → 0.75. Un modelo más
  grande memoriza con más fuerza que el patrón no existe.
- **Sin marca.** El mejor control queda a menos de 0.02 de F. En un recorte sin marcas más chico
  (10%), cambiar de modelo movió G hasta 0.06. Esa es la palanca del ingeniero.

**Con diez veces más datos.** BigF, su corte de clase y su control se entrenaron con el modelo
mediano durante 30,000 pasos.

| Modelo | BLiMP | G pronombres | G modales |
|---|---|---|---|
| BigF | 0.629 | 0.910 | 0.902 |
| control | 0.632 | 0.822 | 0.895 |
| corte de clase | 0.622 | 0.761 | 0.565 |

- **BLiMP sube de 0.57 a 0.63.**
- **Modales: el resultado de atribución es más fuerte a escala.** El corte deja a los modales casi
  en el azar.
- **Pronombres: la brecha baja a 0.06, y nuestras predicciones fallaron.** Con este tamaño de
  datos, casi todo verbo en pasado frecuente ya sigue a cada pronombre en la referencia, lo que
  deja 3–7 ítems no vistos por pronombre, y las lecturas por cabeza se vuelven erráticas. No
  podemos separar la falta de potencia de la prueba de una reducción real. Más datos también traen
  más evidencia indirecta (sujetos sustantivos), como reporta FiCT.

### 5.6 La detección de la herramienta

Los 22 cortes etiquetados de §4.1 se diagnosticaron contra F con 1.50 tokens por palabra.

**Filas de clase.** Se encuentra cada clase eliminada, cada una en su propia rama del árbol, y no se
marca ningún control.

| Corte | Nodo reportado | t (umbral de clase 13.9) |
|---|---|---|
| corte de clase de pronombres y modales | *can will would may could should* (8 cabezas) | 319 |
| | *I you we they i* | 254 |
| corte de clase de posesivos | *your his my their* | 165 |
| corte de clase de números | *two … seven* | 104 |
| corte de clase de verbos base | *get make see take find leave* | 101 |
| corte de cabeza de pronombres y modales | *can will would could should* | 33 |
| corte de cabeza de posesivos | *his their*, exactamente las cabezas quitadas | 85 |
| cortes de cabeza de números y de verbos base | nada | |
| 12 controles aleatorios | nada | |

**Filas de cabeza.** El corte de clase y el corte de cabeza marcan 39 y 13 filas de cabeza; los 12
controles, ninguna.

**Filas ausentes.** Se encuentra cada una de las 8 palabras borradas. En 8 controles aleatorios
nuevos, dos filas se leen ausentes, una en cada uno de dos controles (` luck`, mostrada 0 veces
frente a 7.0 ± 1.8 en los recortes, y `Maybe`).

**Muestras aleatorias de la referencia.** 48 cortes aleatorios de documentos enteros de F, leídos
contra un mismo conjunto fijo de recortes: 2 marcas de cabeza (4%), ninguna fila de clase y ningún
`absent`. alpha es 0.06 por familia.

En la primera versión, con 64 clases de cabezas y un piso de clase de 0.5 nats, las filas de clase
encontraron 4 de 5 clases eliminadas, y se les escaparon los pronombres, con +0.47 nats. Los
posesivos se encontraron solo a través de sus cabezas en falta, porque compartían clase inducida
con *the* y *a*. La primera regla de `absent` se disparó 79–114 veces en controles aleatorios.

### 5.7 Comparación con métricas de corpus y con BLiMP

**Métricas de un solo número.** Sobre los cortes de clase, los cortes de cabeza y ocho controles
aleatorios calculamos: perplejidad bajo el bigrama de F y bajo un bigrama de Wikipedia, KL de
unigramas desde F, MATTR, distinct-2, y el puntaje Vendi de embeddings MiniLM.

- **La prueba planificada (3 desviaciones estándar de los controles) marca como mucho un corte de
  clase por métrica.** Esa prueba fue indulgente: los controles mezclaban tamaños y la
  perplejidad media tenía un valor atípico.
- **A igual tamaño, como lectura exploratoria.**
  - La KL de unigramas y distinct-2 sí registran los dos cortes de clase grandes, que quitaron
    18,126 y 10,315 oraciones (KL +0.019 y +0.011, frente a un ruido de alrededor de 0.001).
  - Se quedan dentro del ruido en los números (1,972 oraciones) y en los verbos base (3,642), que
    el indicador marca con t 104 y 101.
  - Ninguna métrica dice qué patrón cambió.
- **Lo que esto respalda es la localización y la sensibilidad a pérdidas chicas y específicas.** No
  que las otras métricas no vean nada.

**BLiMP.**

- Nuestros modelos pequeño y mediano sacan 0.56–0.58 (azar 0.5), y 0.63 a escala. La concordancia
  sujeto–verbo queda alrededor de 0.52 a escala chica.
- Entre control y corte de clase, BLiMP se mueve como mucho 0.017, mientras que G se mueve
  0.11–0.34. Un solo puntaje sobre 67 paradigmas diluye una pérdida específica de una clase.
- G y BLiMP miden sintaxis distinta. G lee las clases de las posiciones locales, que nuestros
  modelos aprenden bien (0.96–1.00 en pronombres y modales). Los paradigmas de BLiMP dependen sobre
  todo de dependencias más largas.

### 5.8 Sin ajustes arbitrarios

Cada ajuste de la primera versión se derivó de los datos o se midió contra los veredictos sobre los
22 cortes etiquetados de §5.6.

**Granularidad.** Se aprendió un tokenizador sobre F hasta 60,000 fusiones y se cortó en cada
granularidad.

| Tokens/palabra | Fusiones | Clases encontradas (de 5) | Controles con marca de clase (de 12) | Controles con marca de cabeza (de 12) | Palabras borradas |
|---|---|---|---|---|---|
| 1.35 | 30,526 | 5 | 1 | 0 | 8/8 |
| 1.40 | 20,608 | 5 | 0 | 0 | 8/8 |
| 1.50 | 11,381 | 5 | 0 | 0 | 8/8 |
| 1.60 | 7,220 | 5 | 1 | 1 | 8/8 |
| 1.70 | 4,944 | 5 | 0 | 3 | 8/8 |
| 1.80 | 3,548 | 5 | 0 | 0 | 8/8 |

- Los veredictos se mantienen en todo el rango, y las marcas en controles (6 de 144 lecturas)
  están cerca de la tasa de diseño.
- Entre granularidades vecinas, insuficiente y dentro del ruido se intercambian en 26–37 de
  330,000–690,000 patrones. Lo que se mueve es si un patrón raro se lee o no.
- 1.30 no entra en 16 GB: 45,748 tokens para agrupar. Así que el rango no tiene un borde medido en
  ninguno de los dos lados, y el objetivo de la herramienta, 1.50, queda dentro con margen.
- Un presupuesto fijo no significa nada entre corpus: 4,000 fusiones dejan 1.76 tokens por palabra
  en F, mientras que en 30 documentos de recetas las fusiones se agotan en 269, cerca de palabras enteras.
  Por eso la herramienta aprende hasta una granularidad, no hasta un presupuesto, y rechaza un presupuesto fuera del rango.

**Los ajustes de clase.** Se forzó el corte grueso de 8 a 64 clases, y el contexto de 250 a 4,000
tokens (del 51% al 87% de los tokens del texto). En cada valor se encuentran las cinco clases, no
se marca la fila de clase de ningún control, las marcas de cabeza en controles quedan en 0–2 de 12,
y se encuentran todas las palabras borradas. Las filas de clase no dependen en absoluto del corte
grueso: se leen sobre todo el árbol, y el umbral de clase es 13.92 en todos los casos. Una fila de
cabeza es una cabeza seguida de una clase de relleno, así que otro corte le hace otra pregunta a la
misma cabeza; la coincidencia a nivel de patrón con el corte de 12 clases baja del 97% (8 clases)
al 63% (64).

**El umbral y los recortes.** Con 16 recortes y el máximo como umbral, se marcaron 6 de 48 cortes
aleatorios contra un conjunto de recortes, y ninguno con recortes nuevos para cada corte: el umbral
estaba calibrado en promedio sobre conjuntos de recortes, pero el máximo de 16 recortes también
tiene ruido (de 5.6 a 9.6 entre conjuntos). Una reserva de 200 recortes coincidió con la ley Beta
de §3.4 con 16–128 recortes. Con 101 recortes y k = 6, se marcan 2 de 48.

**Lo que reemplazaron.** El piso de diversidad de 0.5 y el margen de 3 sigma habrían necesitado una
recalibración para cada corpus nuevo: cuando todo token pasó a ser relleno, el piso de clase de
0.5 nats no detectó ninguna clase eliminada salvo una, y ningún piso separaba los desplazamientos
de clase que les costaron la sintaxis a los modelos de los que no. El tronco mínimo de 20, barrido
de 2 a 40, no cambió ningún veredicto entre 5 y 40 en F, pero movió ambos umbrales, y no se
trasladaría a un corpus de otro tamaño.

### 5.9 Sin referencia

**La saturación sola no marca una posición empobrecida.** El patrón pronombre → verbo se empobreció
al 5–50% de sus oraciones con los mismos bytes, y cada corpus se leyó por sí solo. La masa
faltante de la posición sube de 0.028 a alrededor de 0.05, pero el nodo mediano de cada corpus da
0.40: cualquier tolerancia que marcara la posición empobrecida marcaría la mayor parte del corpus.
Si se quita por completo, la posición vuelve a dar 0.028, tan saturada como el control.

**Los hermanos sí.** Con la prueba de hermanos de §3.5:

- En `head_cut`, *they* muestra 3 verbos donde una partición aleatoria de su padre da 26.5 (t 9.6,
  umbral 6.3). La herramienta marca *she* y *they*, y ningún pronombre del control.
- **Réplica.** Las dos mitades aleatorias del control, leídas sobre el árbol y las clases del
  propio control para que cada prueba sea la misma prueba, comparten 264 pruebas débiles, de 332 y
  397.
- En la gramática de recetas, 11 de 14 verbos restringidos salen débiles. Los otros tres comparten
  su único ingrediente con su hermano. Ningún verbo sale débil en las recetas diversas.
- En FineWeb la herramienta marca 95 filas de cabeza en `control` y 120 en `head_cut`, en 2.5
  minutos.

**Qué son las marcas.** La mayoría son cabezas que la gramática restringe dentro de su clase de
relleno: *to* toma menos rellenos que las otras preposiciones (524 frente a 871), y los verbos de
reporte (*said, says, means*), menos que otros verbos. Son débiles en el corpus, que es lo que ve un
modelo entrenado con él. Una referencia las lee dentro del ruido, porque el dominio también las
tiene.

**Qué no puede ver.** Empobrecer todos los pronombres a la vez no deja ningún hermano más fuerte: el
nodo de pronombres da 0.94–1.06 de sus rellenos esperados cuando se empobrece al 25% o al 10%, y
`class_cut` muestra su posición 19 veces. En un corpus sintético de sujeto–verbo, *she* y *they*
restringidos a dos verbos forman su propia clase y pierden a su hermano.

**Descartado.** Comparar cada cabeza con toda su clase gruesa, agregar una prueba de con qué
frecuencia una cabeza toma cada clase de relleno, y agrupar cabezas por contexto marcaron
500–2,000 cabezas y 3,000–15,000 filas de clase en el control, entre ellas todos los pronombres.
Sus mitades compartieron el 12–17% de las marcas.

---

## 6. Discusión

**Qué certifica el indicador.**

- Un patrón de clase marcado es una falla del corpus. En nuestros experimentos ninguna
  configuración lo recuperó, y más capacidad lo empeoró. El remedio son datos: texto en que las
  cabezas de la clase tomen otros rellenos de su posición.
- Un corpus sin marcas permite alcanzar la sintaxis de la referencia para su tamaño. Ahí, una
  falla apunta al modelo.
- Esta división del trabajo es el punto práctico. El lado de los datos certifica que los patrones
  están servidos. Encontrar el modelo que los aprenda queda para la ingeniería del modelo.

**Por qué a nivel de clase.**

- Los modelos comparten evidencia entre las cabezas de una clase. Una cabeza con una posición pobre
  la toma prestada de sus hermanos, lo que coincide con los resultados de evidencia indirecta de
  FiCT y del estudio AANN.
- Los filtros realistas empobrecen cabezas, pero rara vez vacían clases. Eso explica por qué, entre
  recortes realistas, la sintaxis de clase siguió al tamaño y al tema y no al puntaje (§5.4).
- Un diagnóstico pensado para la sintaxis debe por lo tanto leer clases, mientras que un
  diagnóstico de qué palabra predecirá el modelo lee cabezas. Leer cada nodo del árbol de cabezas,
  en vez de un solo corte, permite que el reporte nombre el grupo que carga un desplazamiento: toda
  la clase de los modales en un corte, dos posesivos en otro.

**Ajustes a partir de los datos.** Un umbral fijado sobre los corpus en que se desarrolló un método
es una afirmación sobre esos corpus. El umbral del indicador, en cambio, dice para cada corpus
cuánto se aleja, en el peor caso, una muestra aleatoria de la referencia; al lector le queda un
solo número por juzgar, alpha, que es una tolerancia y no una propiedad de los datos. El costo es
tiempo: con 101 recortes, un diagnóstico tarda 4–9 minutos con nuestros tamaños, frente a unos
4 minutos antes.

**Con y sin referencia.**

- Sin referencia, el corpus se juzga por su propia evidencia: una cabeza más débil que sus
  hermanos es débil para un modelo entrenado con él. Esta es la lectura que tiene quien trabaja con
  estos modelos antes de encontrar una referencia, y marca huecos reales (*she* y *they* en
  `head_cut`).
- Una referencia hace dos cosas más. Ve lo que al corpus le falta por completo, o le falta a toda
  una clase, y separa la debilidad del corpus de la del dominio.
- Ambas lecturas son relativas. "Sin marcas" significa tan bueno como una muestra aleatoria de la
  referencia de este tamaño, o como una parte aleatoria del padre de la cabeza; no significa tan
  bueno como la referencia a cualquier tamaño.
- En nuestros datos, un recorte aleatorio de un corpus grande y bien construido ya conserva los
  patrones casi tan bien como cualquier filtro. El papel práctico del indicador es sobre todo
  rechazar recortes y filtros que sesgan la sintaxis.

**Tamaños de efecto.**

- El efecto del daño a las posiciones sobre el modelo completo es pequeño en bits por byte (0.017
  en toda una escalera de dosis de un filtro).
- El efecto por patrón es grande: 1–4.6 nats en cabezas plantadas, 0.13–0.33 de G en eliminaciones
  de clase.
- Por eso hace falta un diagnóstico por patrón: los números del corpus o del modelo completo
  diluyen las pérdidas locales.

---

## 7. Limitaciones

- **Escala.** Los modelos llegan a 6.2M de parámetros, y los corpus a unos 10M de palabras, en
  CPU. En BLiMP los modelos sacan 0.56–0.63. Las afirmaciones tratan de sintaxis local aprendida por
  modelos pequeños. No está probado que la atribución a nivel de clase se mantenga en modelos que
  aprenden sintaxis de largo alcance.
- **Solo sintaxis local.** Cabezas de uno o dos tokens y el relleno siguiente. Las dependencias
  largas (concordancia a distancia, relleno–hueco, ligamiento) quedan fuera de alcance.
- **Un idioma y una fuente natural**: inglés, FineWeb, más una gramática sintética. Los ajustes se
  midieron en FineWeb; que se mantengan en otro dominio es lo esperable por cómo se derivan, pero
  no está demostrado.
- **Clases probadas a fondo.** Tres clases (pronombres, modales, posesivos) entrenadas con
  eliminación; dos más (números, verbos base) revisadas solo del lado del corpus. A escala, el
  resultado de pronombres no se confirmó, porque su prueba perdió potencia.
- **Semillas.** Dos semillas en la mayoría de los experimentos; una en las configuraciones nuevas
  de modelo y en las corridas de escala.
- **El nivel de la prueba.** alpha = 0.06 es por familia. Un corpus aleatorio puede marcarse en
  cualquiera de tres familias, así que su probabilidad de alguna marca es como mucho del 18%;
  medida, fue del 4% (2 de 48). Repartir alpha entre las tres necesitaría unos 300 recortes. En las
  partes plantadas, se marcó una fila de clase en cada una de dos de las siete partes sin plantar.
- **El rango de granularidad** está acotado por lo que pudimos medir: no apareció ningún borde
  entre 1.35 y 1.80, y 1.30 no entra en 16 GB.
- **Sin referencia**, no se ven un patrón que le falte al corpus, una clase mostrada débilmente en
  su conjunto, ni una cabeza que el árbol separe de sus hermanos.
- **D en los plantados.** El ajuste de dosis de §5.2 (caída ≈ −0.68 · D) y la D por cabeza de
  §5.4 vienen de la lectura de tokens de tipo palabra de la primera versión.
- **Tamaño del modelo.** En la gramática de recetas, la diversidad que necesita un modelo creció
  con su tamaño. El indicador dice dónde un corpus se aparta de su referencia; cuánto apartamiento
  tolera un modelo dado no es parte de él.
- **Sintaxis, no semántica.** Un corpus puede pasar y aun así fallarle al modelo por tema o
  significado: un corpus de tema estrecho perjudicó a los modelos en todos los documentos sin que
  se marcara ninguna posición.

---

## 8. Conclusión

Preguntamos si un corpus puede revisarse, antes de entrenar, en busca de los patrones sintácticos
locales que su dominio necesita, de modo que una falla de generalización se pueda atribuir al
corpus o al modelo. La respuesta, dentro de los límites anteriores, es sí.

Un indicador que compara los rellenos de cada posición con muestras aleatorias de una referencia,
en cada nodo de un árbol inducido de cabezas, marca fallas del corpus que ninguna configuración de
modelo probada recupera. Donde no marca nada, un modelo alcanza la sintaxis de la referencia.
Ninguno de sus ajustes se fija a mano: la granularidad, el tronco de lectura y el número de
muestras salen de los datos, y el nivel de la prueba se declara en cada reporte. Sin referencia, el
mismo árbol permite juzgar un corpus por su propia evidencia. En el camino encontramos que los
modelos aprenden la sintaxis local sobre todo a nivel de clase, lo que explica tanto por qué los
filtros de datos realistas rara vez rompen la sintaxis como por qué un diagnóstico debe leer
clases. La herramienta, el código y cada predicción prerregistrada, incluidas las fallidas, se
publican junto con el artículo.

**Trabajo futuro:**

- modelos y corpus más grandes, con GPU;
- dependencias de largo alcance;
- más idiomas y dominios, para probar que los ajustes derivados se trasladan;
- la tolerancia de un tamaño de modelo dado a un apartamiento marcado;
- una contraparte semántica para los dominios de conocimiento.

---

## Apéndice A. Predicciones que fallaron

Se conservan tal como se escribieron en los documentos de prerregistro.

**Indicador y modelos (primera versión).**

- Gramática de recetas: los pares mínimos "cut the slowly" no se quedaron cerca del 100% (0.25 con
  un relleno). Las predicciones de tamaño de corpus fallaron con 500 documentos.
- Plantados: una cabeza de comparación se movió −0.51 nats. La pérdida sobre todo el texto subió
  0.03 bits por byte (se predijo < 0.02). Con k = 32 todavía se detectó a menudo (12/20).
- Corpus de tema estrecho: las cabezas marcadas no perdieron más. La pérdida fue temática y en
  todos los documentos.
- Selección de subconjuntos: las marcas naturales suaves (diversidad 0.26–0.50) no le costaron al
  modelo. En r10, Spearman(z, caída) fue 0.10 / 0.12 (umbral 0.15).
- Dosis: el extremo suave costó 0.38–0.57 nats (se predijo ≤ 0.3). El umbral se cruzó en 0.64.
- Divergencia: en los plantados, D no ordenó las cabezas mejor que la diversidad. Su primer corte,
  κ, falló por un piso de tamaño de muestra.
- Elección de recortes: sobre diez recortes, Spearman(puntaje, bits por byte) fue 0.55 / 0.45 (se
  predijo ≥ 0.5 en ambas semillas). Con la lectura actual se cumple (0.65 / 0.55).
- Escaleras: la escalera que suma documentos dio Spearman 0.60. La lectura por cabeza falló con
  dosis del 25%.
- Evaluación sintáctica de recortes realistas: fallaron las seis predicciones. Esto llevó a la
  hipótesis de nivel de clase.
- Patrones de clase: P2 −0.086 (umbral −0.10), con una primera prueba de modales inválida. P4: el
  ruido entre modelos superó el umbral.
- Réplica: el corte por cabeza de posesivos costó el 57% (umbral 50%).
- Herramienta: no detectó la clase de pronombres (+0.47 frente a un piso de 0.5). La primera regla
  de `absent` dio 79–114 falsas alarmas. Un control dio 5 con m/2.
- BLiMP: los modelos de F sacaron menos de 0.60.
- Escala: la brecha de pronombres fue 0.06 (umbral 0.10), y el control quedó 0.09 por debajo de la
  referencia.

**Ajustes a partir de los datos.**

- Granularidad: se predijo que granularidades vecinas coincidirían en el 90% de los patrones;
  coincidieron en el 85–94%, y casi toda la diferencia era si un patrón se lee o no. Con un tronco
  de 20, los números y los verbos base no se detectaron con 1.35 (se predijo que se encontrarían
  en todo el barrido).
- Piso de clase: se predijo que cada clase cuya eliminación les costó la sintaxis a los modelos
  daría un exceso mayor que todas las que no. Falló: con 64 clases de cabezas sobre todo token,
  pronombres y posesivos compartían una clase con *the* y *a*, y la movieron +0.045 y +0.007. Esto
  llevó a las filas de clase en cada nodo.
- El umbral de 16 recortes: se marcaron 6 de 48 cortes aleatorios contra un conjunto de recortes
  (diseño: alrededor de 1 en 17). Esto llevó a alpha y a los 101 recortes.
- Sin referencia: la saturación (masa faltante) no marcó una posición empobrecida. Tres
  agrupaciones de cabezas dieron marcas que las mitades del corpus no compartían.
- Selección de subconjuntos, repetida: se predijo que ambos filtros marcarían al menos 10 filas;
  marcaron 2 y 1.

## Apéndice B. Reproducibilidad

Cada experimento tiene una carpeta en `experiments/` con su script, sus resultados (JSON) y un
README. `docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` contiene los diseños, predicciones y
resultados de la primera versión, en orden, y `docs/superpowers/specs/2026-10-11-settings-from-data-design.md`, la
derivación de cada ajuste. Los resultados de la primera versión se conservan junto a las nuevas
corridas (`results/three_sigma/` en `experiments/subset_selection`, las tablas del primer margen en
`experiments/planted_gap`). `corpus-diagnose` y la página de Streamlit están en
`src/corpus_generalization/`. Todo corre en CPU; un diagnóstico tarda 4–9 minutos con nuestros
tamaños, después de leer la referencia una vez. Los tiempos de cada corrida están en su README.

---

## Referencias

**Verificadas en su fuente para este borrador:**

- Patil, A., Jumelet, J., Chiu, Y. Y., Lapastora, A., Shen, P., Wang, L., Willrich, C., &
  Steinert-Threlkeld, S. (2024). Filtered Corpus Training (FiCT) Shows that Language Models Can
  Generalize from Indirect Evidence. *TACL* 12, 1597–1615.
- Luo, J., & Steinert-Threlkeld, S. (2026). Exposure is Optional: Learning Unlike Coordination in
  Language Models. arXiv:2607.20251.
- Misra, K., & Mahowald, K. (2024). Language Models Learn Rare Phenomena from Less Rare Phenomena:
  The Case of the Missing AANNs. *EMNLP 2024*.
- Wei, J., Garrette, D., Linzen, T., & Pavlick, E. (2021). Frequency Effects on Syntactic Rule
  Learning in Transformers. *EMNLP 2021*, 932–948.
- Miranda, B., Lee, A., Sundar, S., Casasola, A., Schaeffer, R., Obbad, E., & Koyejo, S. (2023).
  Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural
  Language Data. arXiv:2306.13840.
- DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs (2026). *ICLR 2026*.
  arXiv:2603.19688.
- Wettig, A., et al. (2024). QuRating: Selecting High-Quality Data for Training Language Models.
  arXiv:2402.09739.
- Xie, S. M., Santurkar, S., Ma, T., & Liang, P. (2023). Data Selection for Language Models via
  Importance Resampling. arXiv:2302.03169.
- Albalak, A., et al. (2024). A Survey on Data Selection for Language Models. arXiv:2402.16827.
- Guo, Y., et al. (2024). Benchmarking Linguistic Diversity of Large Language Models.
  arXiv:2412.10271.
- Baayen, R. H. (2009). Corpus linguistics in morphology: morphological productivity. In
  *Corpus Linguistics: An International Handbook*.
- Marvin, R., & Linzen, T. (2018). Targeted Syntactic Evaluation of Language Models.
  arXiv:1808.09031.
- Hu, J., Gauthier, J., Qian, P., Wilcox, E., & Levy, R. (2020). A Systematic Assessment of
  Syntactic Generalization in Neural Language Models. arXiv:2005.03692.
- Warstadt, A., et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English.
  *TACL* 8.

**Recomendadas, por verificar antes del envío:**

- Tarone, R. E. (1990). A modified Bonferroni method for discrete data. *Biometrics* 46.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing*. Wiley.
- Linzen, T., Dupoux, E., & Goldberg, Y. (2016). Assessing the ability of LSTMs to learn
  syntax-sensitive dependencies. *TACL*.
- Wenzek, G., et al. (2020). CCNet: Extracting High Quality Monolingual Datasets from Web Crawl
  Data. *LREC*.
- Penedo, G., et al. (2024). The FineWeb Datasets.
- Warstadt, A., et al. (2023). Findings of the BabyLM Challenge.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with
  Subword Units. *ACL*.
- Good, I. J. (1953). The population frequencies of species and the estimation of population
  parameters. *Biometrika*.
- Witten, I. H., & Bell, T. C. (1991). The zero-frequency problem. *IEEE Trans. Inf. Theory*.
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*.
- Honnibal, M., et al. (2020). spaCy: Industrial-strength NLP.
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: the moving-average
  type–token ratio (MATTR). *J. Quantitative Linguistics*.
- Li, J., et al. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models
  (distinct-n). *NAACL*.
- Friedman, D., & Dieng, A. B. (2023). The Vendi Score: A Diversity Evaluation Metric for Machine
  Learning. *TMLR*.
- Lake, B., & Baroni, M. (2018). Generalization without systematicity (SCAN). *ICML*.
- Kim, N., & Linzen, T. (2020). COGS: A Compositional Generalization Challenge. *EMNLP*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
