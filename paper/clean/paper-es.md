# ¿Culpa de los datos o del modelo? Diagnóstico de la sintaxis de un corpus de dominio antes de entrenar

**Marco Fura Mendoza**\
Pontificia Universidad Católica de Chile\
mjfura27@gmail.com

*Borrador, octubre de 2026. No distribuir. Versión en español; la versión de referencia es la
inglesa.*

---

## Resumen

Cuando un modelo de lenguaje no generaliza un patrón de su dominio, la causa puede estar en el
modelo o en el corpus, y las medidas existentes de calidad de datos no distinguen entre ambas.
Un dominio de conocimiento, como las recetas, es un subconjunto acotado de los patrones de una
lengua; aquí abordamos su capa sintáctica.
Proponemos un indicador de suficiencia sintáctica local calculado solo sobre el corpus. Lee cada
patrón sintáctico local del corpus en dos niveles, una cabeza seguida de una clase gramatical
(*she* → verbo) y una clase de cabezas inducida del corpus (pronombre → verbo), y compara la
variedad de sus rellenos con muestras aleatorias del mismo tamaño de un corpus de referencia; sin
referencia, cada cabeza se compara con sus hermanos en el propio corpus. Ningún umbral se ajusta a
mano: la evidencia mínima que se lee y el número de muestras aleatorias salen de los datos, la
granularidad del tokenizador se fija dentro de un rango medido en cinco dominios, y se muestra que
los demás ajustes no cambian los veredictos, de modo que solo queda el nivel de la prueba. Validamos el indicador con
intervenciones prerregistradas sobre el corpus y modelos entrenados. En una gramática sintética
predice qué patrones generaliza un modelo (Spearman 0.94–0.98), mientras que la pérdida en texto no
visto ordena los corpus al revés. En FineWeb encuentra huecos plantados sin falsas alarmas a nivel
de cabeza, en cabezas donde los modelos pierden 1–4.6 nats en la palabra siguiente verdadera. Quitarle un patrón
a toda una clase se lo quita a cada miembro, mientras que quitárselo a dos cabezas les cuesta solo
entre el 25 y el 57% de eso: la sintaxis local se aprende sobre todo a nivel de clase. Cuando el
indicador marca un patrón de clase, ninguna configuración de modelo que probamos lo recupera;
cuando no marca nada, modelos de hasta 6.2M de parámetros alcanzan la sintaxis de la referencia
con una diferencia de 0.02. A
diferencia de las métricas de corpus de un solo número, el indicador localiza el patrón dañado. Su
alcance es la sintaxis de adyacencia, validada con modelos de hasta 6.2M de parámetros.

**Palabras clave:** dominios de conocimiento; corpus de dominio específico; suficiencia
sintáctica; diagnóstico de datos de entrenamiento; generalización sintáctica; diversidad de
rellenos; modelos de lenguaje

---

## 1. Introducción

Quien entrena un modelo de lenguaje con un corpus de dominio y ve que no generaliza enfrenta una
pregunta de diagnóstico antes que cualquier pregunta de ingeniería. ¿Al corpus le falta algo, o
el modelo no es el adecuado para él? Las dos respuestas piden acciones opuestas. Recolectar o
rebalancear datos no puede arreglar un modelo demasiado pequeño o mal entrenado, y ajustar la
arquitectura no puede arreglar un corpus que nunca muestra un patrón con suficiente variedad. Hoy
esa pregunta se responde, cuando se responde, entrenando e inspeccionando.

Planteamos la pregunta a través de la noción de **dominio de conocimiento**: un subconjunto
acotado de los patrones sintácticos y semánticos de un idioma, como las recetas, que son sobre
todo acciones imperativas sobre ingredientes. Este artículo trata solo la capa sintáctica de un
dominio. Por sintaxis entendemos el marco bien formado de una oración, sea verdadera o no. "Corta
la leche" es sintácticamente correcta; que no tenga sentido es asunto de la semántica y queda
fuera de nuestro alcance. Nuestra premisa sigue una observación de larga data en lingüística de
corpus (las medidas de productividad de Baayen, 2009) y en el aprendizaje de modelos de lenguaje
(Wei et al., 2021; Misra & Mahowald, 2024): un patrón se aprende como patrón, y no como palabra
memorizada, cuando el corpus muestra su posición con suficientes rellenos distintos. Decimos que
un corpus es **sintácticamente suficiente** para su dominio cuando muestra de este modo cada
patrón local del dominio.

Las herramientas existentes no miden esto. Los métodos de calidad y selección de datos, como el
filtrado por perplejidad al estilo CCNet (Wenzek et al., 2020), DSIR, QuRating y el coeficiente
de diversidad, dan un número por documento o dataset, pensado para la perplejidad o el desempeño
en tareas, y un solo puntaje no puede decir qué patrón falta. La pérdida en texto no visto puede
engañar: en una gramática controlada mostramos que ordena los corpus al revés de la
generalización sintáctica, porque un corpus restringido también lo está en su partición de
texto no visto y, por lo tanto, parece más predecible. Por último, el entrenamiento con corpus
filtrados (FiCT; Patil et al., 2024) y estudios afines quitan una construcción y prueban qué
aprende el modelo entrenado. Estos estudios son posteriores al entrenamiento por diseño, y su
mensaje principal, que los modelos generalizan desde evidencia indirecta, hace que la pregunta
de qué eliminaciones importan sea real.

Un diagnóstico de este tipo es tan confiable como sus ajustes. Un umbral ajustado sobre los
corpus en que se desarrolló el método puede producir, en un corpus nuevo, marcas que son
artefactos del umbral. Por eso exigimos que cada ajuste del indicador se derive de los datos o
se demuestre que no cambia sus veredictos, y reportamos las mediciones (§5.8).

Nuestras contribuciones son las siguientes:

1. **Un indicador calculado solo sobre el corpus.** Lee patrones sintácticos locales a nivel de
   cabeza y en cada nodo de un árbol inducido de cabezas, y los compara con muestras aleatorias
   de una referencia del mismo tamaño o, sin ella, con las cabezas hermanas del corpus (§3);
   viene empaquetado como herramienta de línea de comandos y como interfaz de usuario.
2. **Ajustes a partir de los datos.** Un rango medido de granularidad del tokenizador, un nivel
   de prueba que fija el número de muestras aleatorias y un tronco de lectura elegido por el
   criterio de Tarone reemplazan todo umbral fijado a mano (§3.1, §3.4, §5.8).
3. **Una validación causal mediante intervenciones sobre el corpus** en una gramática sintética
   y en texto natural, que abarca huecos plantados, dosis–respuesta, filtros realistas, y la
   eliminación de un patrón a una cabeza frente a toda su clase (§5).
4. **Un hallazgo sobre el nivel en que se aprende la sintaxis.** Una cabeza que pierde su
   posición conserva su sintaxis mientras su clase conserve el patrón, mientras que una clase que
   pierde el patrón pierde la sintaxis de todos sus miembros (§5.4).
5. **Una prueba de atribución.** Un patrón de clase marcado es una falla del corpus que ninguna
   configuración de modelo probada arregla, y un corpus sin marcas permite a modelos pequeños
   alcanzar la sintaxis de la referencia (§5.5); esta es la garantía que necesita quien trabaja con estos modelos.
6. **Comparaciones con métricas de corpus de un solo número y con BLiMP**, con un recuento
   honesto de dónde ven lo mismo y dónde no (§5.7), junto con una lista de todas las
   predicciones fallidas (Apéndice A).

---

## 2. Trabajo relacionado

*Calidad y selección de datos para preentrenamiento.* El filtrado por perplejidad contra un
modelo de lenguaje de referencia, como en el pipeline CCNet, es un filtro de calidad estándar.
DSIR (Xie et al., 2023) selecciona datos por remuestreo de importancia sobre rasgos de n-gramas
con hash, y QuRating (Wettig et al., 2024) usa juicios de un modelo de lenguaje sobre calidad de
escritura, experticia y valor educativo. El coeficiente de diversidad (Miranda et al., 2023) mide
la variabilidad de un dataset mediante embeddings Task2Vec. Predictores de influencia de los
datos sin entrenamiento, como DataProphet (Qi et al., 2026), ordenan datasets de supervisión
antes de entrenar. Albalak et al. (2024) revisan el área. Todos estos métodos califican
documentos o datasets como un todo. Ninguno reporta qué patrones sintácticos sirve mal un corpus,
y ninguno separa una falla del corpus de una falla del modelo. Encontramos que los filtros
basados en perplejidad son los únicos recortes de un corpus que el indicador marca (§5.3).

*Aprender sintaxis de corpus manipulados.* Wei et al. (2021) muestran que BERT aplica la
concordancia sujeto–verbo a pares nunca vistos juntos, pero tiene problemas con ítems poco
frecuentes y con sesgos fuertes. FiCT (Patil et al., 2024) quita una construcción del corpus de
entrenamiento y encuentra que tanto LSTMs como transformers la generalizan desde evidencia
indirecta; Luo & Steinert-Threlkeld (2026) extienden este resultado a la coordinación de
categorías distintas. Misra & Mahowald (2024) muestran que una construcción rara (AANN, "a
beautiful five days") se aprende de construcciones relacionadas más frecuentes, y se aprende
mejor con más variabilidad en el input. Nuestros resultados coinciden con estos hallazgos a
nivel de cabeza: un pronombre que pierde su posición de verbo conserva su sintaxis gracias a sus
hermanos. Agregamos el nivel de clase, donde la eliminación sí impide el aprendizaje, y un
indicador que dice antes de entrenar qué eliminaciones importan.

*Medidas de productividad y diversidad.* Las medidas de productividad de Baayen (2009) usan los
hápax legómenos para estimar cuán abierta está una categoría morfológica. La masa faltante de
Good–Turing (Good, 1953), singletons sobre tronco, aplica la misma idea a una posición
sintáctica; mostramos que por sí sola no marca una posición empobrecida (§5.9). Las medidas de
diversidad léxica, sintáctica y semántica de las salidas de un modelo (Guo et al., 2025) evalúan
texto generado y no corpus de entrenamiento, y no están ligadas a lo que un modelo aprende.

*Pruebas múltiples con estadísticos discretos.* El indicador prueba decenas de miles de patrones
a la vez. Su umbral es el estadístico máximo sobre muestras aleatorias, un umbral de error por
familia basado en remuestreo (Westfall & Young, 1993), y los patrones demasiado raros para
marcarse alguna vez quedan fuera de la familia, siguiendo a Tarone (1990).

*Evaluación sintáctica dirigida.* Los pares mínimos (Linzen et al., 2016; Marvin & Linzen, 2018),
la evaluación sistemática de Hu et al. (2020) y BLiMP (Warstadt et al., 2020) evalúan modelos ya
entrenados. Nuestra lectura del lado del modelo es un puntaje de pares mínimos construido para
excluir la semántica: rellenos válidos para la clase y nunca vistos compiten contra rellenos
agramaticales de frecuencia equivalente, con clases tomadas de un etiquetador externo (§4.3).
También reportamos BLiMP.

---

## 3. El indicador

La Figura 1 resume el indicador; las subsecciones siguientes definen cada paso.

![**Figura 1.** El indicador. El corpus se tokeniza a 1.50 tokens por palabra y sus tokens se agrupan en un único árbol: cada nodo es una clase de cabezas, y un corte da las clases de relleno. Cada patrón de cabeza y de clase se lee (a) frente a 101 muestras aleatorias de una referencia del mismo tamaño, con un umbral tomado de los máximos, dejando uno fuera, de las propias muestras al nivel α, o (b) sin referencia, frente a particiones aleatorias del padre de la cabeza en el árbol. El recuadro es esquemático.](figures/fig1_method.png){width=100%}


### 3.1 Unidades: tokens, transiciones, clases

El corpus se tokeniza con un BPE a nivel de byte (Sennrich et al., 2016) aprendido sobre el texto
del dominio, que es el texto de la referencia o, sin referencia, el del propio corpus. El
tokenizador se entrena hasta las menos fusiones que dejan 1.50 tokens por palabra; en el conjunto
de FineWeb esto son 11,381 fusiones. Medimos que los veredictos se sostienen entre 1.35 y 1.70
tokens por palabra en cinco dominios, con dominios de al menos 14.4 MB, salvo cuando los usos de
una palabra borrada están en pocos documentos (§5.8). Todo token participa en el análisis: cualquier token que aparezca en
al menos dos documentos puede ser cabeza o relleno, sea una palabra, una parte de palabra,
puntuación o un número.

Una transición es el último token de la cabeza seguido del relleno. La llamamos *sintáctica*
cuando el relleno empieza un nuevo pre-token del tokenizador (una palabra nueva, una secuencia
de puntuación, un número o una contracción), y *léxica* cuando el relleno continúa la palabra
del token anterior (` model` → `s`). Como ninguna fusión cruza un límite de pre-token, esta regla
es exacta. Los dos tipos se cuentan como patrones separados, y las filas sintácticas forman el
diagnóstico.

Las clases provienen de un solo árbol. Cada token se describe por las distribuciones de sus
vecinos a la izquierda y a la derecha sobre los 1,000 tokens más frecuentes. Las filas se toman
en raíz cuadrada, para que la distancia euclídea sea la distancia de Hellinger, y se agrupan con
el método de Ward (1963) en un solo árbol. Las clases de relleno se obtienen cortando el árbol en
el mayor salto de altura de fusión entre 8 y 64 clases, lo que da 12 clases en FineWeb. En la
gramática de recetas este corte recupera exactamente las categorías de la gramática. En texto
natural las clases son posiciones en la oración más que categorías gramaticales (§5.8): salen los
sustantivos y los verbos, mientras que los determinantes, los posesivos y los adjetivos comparten la
clase de lo que va antes de un sustantivo. Un corte
más fino empieza a agrupar los rellenos por significado (lo que se vierte frente a lo que se
pica), lo que metería semántica en una medida sintáctica. Las clases de cabezas, en cambio, no
requieren corte: cada nodo interno del mismo árbol es una clase de cabezas, y los patrones de
clase se leen en cada nodo (§3.2).

### 3.2 Patrones

Un patrón de cabeza es una cabeza de uno o dos tokens seguida de un relleno de la clase Y, junto
con su tipo de transición. Las apariciones se cuentan en contextos distintos, definidos por el
token anterior a la cabeza y el posterior al relleno, de modo que las copias exactas no suman.
Llamamos tronco al número de contextos distintos de un patrón. Un patrón de clase es un nodo del
árbol de cabezas junto con los rellenos que siguen a cualquiera de sus cabezas de un token,
reunidos.

### 3.3 Lecturas

La primera lectura es la diversidad, definida como rellenos / rellenos esperados, donde rellenos
esperados = Σ_y [1 − (1 − p_y)^n], con n el tronco y p_y las proporciones de clase del corpus.
Compara cabezas de distinta frecuencia contra extracciones libres de la clase.

La segunda lectura es la divergencia D = KL(R ‖ S), en nats. Aquí R es la distribución de
rellenos de la posición en la referencia, y S es la del corpus, suavizada con Witten–Bell
(Witten & Bell, 1991) hacia el unigrama de rellenos del corpus, sumando un conteo por cada
relleno de la referencia. D tiene las unidades de la pérdida: un modelo que aprendió S y se
evalúa en texto de R pierde en promedio KL(R ‖ S) nats por aparición de la posición. En los
huecos plantados, la caída en la log-probabilidad del modelo es de alrededor de −0.72 · D
(R² = 0.78, §5.2).

Toda lectura depende del tamaño del corpus, así que un corpus más pequeño siempre sale peor. Por
eso comparamos cada corpus con muestras aleatorias del mismo tamaño: documentos
enteros de la referencia, del mismo tamaño en bytes. La referencia se tokeniza y clasifica una
sola vez y se reutiliza para cada corpus que se diagnostica contra ella.

### 3.4 Veredictos con referencia

El t de cada patrón es cuánto se aleja el corpus de la media de las muestras, en unidades de la
dispersión de las muestras. La lectura es la diversidad en las filas de cabeza, el tronco en
*absent* y D en las filas de clase. La dispersión incluye el redondeo de la propia lectura, ya
que la diversidad se mueve de a un relleno y un tronco de a uno: sd' = √(sd² + step²/12).

El umbral sale de las muestras. Cada muestra, por turno, se deja afuera y se lee contra las
demás, y se registra su mayor t sobre todos los patrones. Un corpus se marca solo donde se aleja
más que el k-ésimo mayor de esos máximos, de modo que una muestra aleatoria de la referencia pasa
el umbral con probabilidad de alrededor de k / (n + 1).

El único ajuste que declara el usuario es alpha, la probabilidad de que una muestra aleatoria de
la referencia se marque en una familia de veredictos. Por defecto es 0.06 por familia, siendo las
familias las filas de cabeza, *absent* y las filas de clase. Alpha es el nivel de la prueba, una
tolerancia y no una propiedad de los datos. El número de muestras se deduce de alpha. El k-ésimo
mayor de n máximos de las muestras también tiene ruido: entre conjuntos de muestras, la
probabilidad de pasarlo sigue una Beta(k, n + 1 − k). La herramienta usa la menor cantidad de
muestras que mantiene esa probabilidad entre alpha/2 y 1.5 · alpha en el 80% de los conjuntos de
muestras, lo que da 101 muestras y k = 6 con alpha 0.06. El tronco de lectura también sale de
las muestras, por el criterio de Tarone. Un patrón demasiado raro para marcarse aun si el corpus
mostrara un solo relleno no aporta ningún veredicto, y su ruido sube el umbral de todos los demás
patrones. El tronco de lectura mínimo es, por lo tanto, el que deja más patrones que podrían
marcarse.

Una fila de cabeza es insuficiente (*under-shown*) cuando su diversidad queda bajo la media de
las muestras por más que el umbral. Es ausente (*absent*) cuando su tronco queda bajo el tronco
medio de las muestras por más que el umbral de *absent*: una muestra aleatoria de este tamaño
mostraría el patrón, y el corpus muestra demasiado poco de él. En los demás casos está dentro
del ruido (*within-noise*), y bajo el tronco de lectura es no leída (*unread*).

Una fila de clase lee D sobre los rellenos reunidos del nodo frente a las muestras, y un nodo es
insuficiente (*under-shown*) cuando su t supera el umbral de clase. De cada rama se reporta el
nodo con el t más alto, de modo que el reporte nombra el grupo de cabezas más pequeño que carga
el desplazamiento. Cada fila de clase lista además sus *cabezas en falta* (*short heads*), los
miembros cuyas propias filas son insuficientes o ausentes.

### 3.5 Veredictos sin referencia

Un patrón que el corpus muestra débilmente es débil para un modelo entrenado con él, haga lo que
haga el dominio. Un corpus solo puede revelarlo comparando una cabeza con sus hermanos. Cada hijo
de un nodo del árbol de cabezas, sea una cabeza o una clase de cabezas, se compara con su padre,
por clase de relleno y tipo de transición. La hipótesis nula pregunta cuántos rellenos distintos
mostraría este hijo si las apariciones de esa clase en el padre se hubieran repartido al azar
entre sus dos hijos. Para responderla se sortean 101 particiones aleatorias de forma exacta, sin
reemplazo, con el tronco del hijo. El umbral se construye a partir de las particiones como en
§3.4, con el mismo alpha, y solo se leen las pruebas que un solo relleno podría hacer fallar.

Se aplican dos restricciones más. Solo se prueban los hijos de un nodo que está dentro de una
clase de relleno: por encima del corte de clases, los dos lados de una partición son categorías
distintas (`,`/`the`/`to` frente a `.`/`:`), y toda prueba así salía débil. Las cabezas de dos
tokens se ubican por su primer token.

Un patrón es débil (*weak*) cuando queda más allá del umbral, y raro (*rare*) cuando se muestra
menos veces que cualquier posición que la prueba pueda leer. En los demás casos está dentro del
ruido (*within-noise*), o no leído (*unread*) cuando no tiene hermano contra el cual leerlo.

Tres casos son invisibles sin referencia. Un patrón que al corpus le falta por completo, y una
clase mostrada débilmente en su conjunto, no dejan ningún hermano más fuerte con el cual
comparar. Una cabeza tan restringida que el árbol la separa de sus hermanos pierde a su hermano.
El reporte declara estos límites y recomienda una referencia del mismo dominio. Una referencia
además separa la debilidad propia del corpus de la del dominio: la mayoría de los patrones
débiles de un corpus natural reflejan su gramática (` to` frente a las otras preposiciones,
verbos de reporte), y una referencia los lee dentro del ruido.

### 3.6 La herramienta

`corpus-diagnose` (Python) recibe un corpus y, opcionalmente, una referencia. Escribe reportes en
JSON y Markdown que abarcan resumen, advertencias, clases, filas de cabeza, filas de clase y
configuración, incluidos alpha y el número de muestras. Cuando no se da un presupuesto de
fusiones, aprende el tokenizador del dominio hasta 1.50 tokens por palabra, retomando el BPE en
rondas que se duplican. Un presupuesto dado con `--merges N` se revisa antes del diagnóstico:
fuera de 1.35–1.70 tokens por palabra, el comando se niega y nombra la granularidad alcanzada,
salvo que se pase `--force`. Avisa cuando el dominio, la referencia o si no el corpus, tiene menos
de 14.4 MB (§5.8). Una página de Streamlit corre el mismo diagnóstico sobre cualquier
dataset del catálogo.

---


## 4. Diseño experimental

### 4.1 Corpus

Usamos una gramática sintética de recetas, texto natural de FineWeb y un conjunto de intervenciones
sobre este último.

La gramática de recetas es una gramática probabilística con léxico cerrado (verbos imperativos,
ingredientes contables, adjetivos, adverbios, utensilios) y cuatro reglas de oración. Una sola
perilla cambia con cuántos ingredientes aparece cada verbo (1, 2, 4, 8, 16, 20, 24 o todos),
mientras cada ingrediente sigue apareciendo en otras oraciones. Las combinaciones verbo–ingrediente
reservadas nunca se ven y sirven como ítems de prueba.

Nuestro texto natural proviene de FineWeb (Penedo et al., 2024). El conjunto principal F consta de
4,600 documentos (14.4 MB, ≈2.4M de palabras) de los primeros 5,000 documentos
descargados en streaming; los otros 400 documentos se reservan para evaluación. Una muestra de
6.5 MB, dividida en 8 partes, se usa para los plantados. BigF son los primeros 20,000 documentos
menos el conjunto de evaluación (57.8 MB, ≈10M de palabras).

Las intervenciones son de cuatro tipos. En los plantados, los seguidores de ciertas cabezas se
reemplazan por uno de sus k rellenos más frecuentes, o se quitan las oraciones en que toman otro
relleno. Los recortes de documentos del 10–50% de F son aleatorios o se hacen con filtros de
perplejidad o filtros de largo. Los recortes a nivel de oración sobre un recorte del 50% de F (r50)
quitan toda oración en que una cabeza de la lista va seguida inmediatamente de una clase etiquetada
por spaCy (Honnibal et al., 2020); se hacen a nivel de clase, a nivel de cabeza (solo dos miembros)
y como controles aleatorios, todos con el mismo tamaño en bytes. Por último, los *cortes
etiquetados* para la herramienta comprenden los cortes de clase y de cabeza de pronombres y
modales, posesivos, números y verbos base (8 cortes), 12 controles aleatorios, y dos cortes que
borran cuatro palabras cada uno.

### 4.2 Modelos

El modelo es un decodificador con la arquitectura de GPT-2 (Radford et al., 2019) y cabeza atada,
entrenado en CPU, en cuatro configuraciones:

- **pequeño:** 2 capas, 128 de ancho, 4 cabezas, contexto 128, 1.9M de parámetros, 3,000 pasos de
  32 × 128 tokens, tasa de aprendizaje 1e-3;
- **mediano:** 4 capas, 256 de ancho, 6.2M de parámetros, tasa de aprendizaje 5e-4;
- **largo:** el pequeño, con 9,000 pasos;
- **escala:** el mediano, 30,000 pasos sobre BigF, unas 5 horas por corrida.

Cada corpus aprende su propio BPE para los modelos (11,563 fusiones en FineWeb, un presupuesto
fijo; el indicador aprende el suyo, 11,381 fusiones a 1.50 tokens por palabra, §3.1), y el mejor checkpoint se elige con
una fracción de validación. Usamos las semillas 0 y 1, salvo que se indique otra cosa. En la gramática de recetas, cuyo tokenizador se detiene en 1,000 fusiones, los modelos pequeño y mediano tienen 0.5M y 3.3M de parámetros.

### 4.3 Lecturas del lado del modelo

Leemos los modelos entrenados de cuatro maneras. La *caída* de una cabeza es la media, sobre sus
apariciones de evaluación, de log p_modelo(palabra siguiente verdadera) − log p_modelo de
referencia, en nats. Es una lectura a nivel de relleno y, por lo tanto, mezcla sintaxis con
semántica y tema.

En la gramática de recetas, la *razón de generalización* es la probabilidad que el modelo da,
después de "⟨bos⟩ cut the", a los ingredientes nunca vistos con ese verbo, relativa a su
proporción; 1 significa que el patrón se aprendió y 0 que se memorizó.

En texto natural, *G* se lee después de un prefijo real de evaluación que termina en la cabeza h.
Es la proporción de comparaciones en que el modelo prefiere una palabra de la clase de la posición
de h que nunca sigue a h en la referencia, frente a una palabra de una clase agramatical, de
frecuencia logarítmica equivalente. La verdad no cuenta: después de "she", cualquier verbo en
pasado es válido. Las clases vienen de spaCy (`en_core_web_sm`), nunca del indicador. Los
contrastes gramaticales son:

- pronombres → verbos en pasado (VBD), contra sustantivos y adjetivos;
- modales → verbos en forma base (VB), contra sustantivos y adjetivos;
- posesivos → sustantivos, contra adverbios y verbos VBD/VBZ.

Por último, puntuamos *BLiMP* (67 paradigmas × 1,000 pares) por log-probabilidad total.

### 4.4 Protocolo

El diseño, las predicciones y los umbrales de cada experimento se escribieron en los documentos de
diseño antes de correrlo. Los parámetros de construcción solo se cambiaron a partir de conteos del
corpus, antes de puntuar cualquier modelo, y cada cambio quedó registrado. Las predicciones
fallidas se conservan y se listan en el Apéndice A.

El indicador se construyó primero con umbrales fijados a mano (un piso de diversidad de 0.5, un
margen de 3 sigma sobre 8 muestras, un tronco mínimo de 20, un piso de clase de 0.5 nats). Estos se
reemplazaron por los ajustes de §3.4–3.5 (§5.8), y cada experimento del lado del corpus se volvió a
correr con la herramienta final. Los modelos no se reentrenaron, porque los corpus son los mismos y
solo cambió su lectura. Los resultados de la versión anterior se reúnen en el Apéndice B.

---

## 5. Resultados

### 5.1 Las medidas en texto no visto no ven la insuficiencia sintáctica (gramática de recetas)

| Medida | Resultado |
|---|---|
| Cobertura por verbo vs. razón de generalización | Spearman 0.891 (una semilla); 0.940–0.975 con 3 semillas × 2 tamaños |
| Razón media, todos los ingredientes vs. uno por verbo | 0.909 vs. 0.011 |
| Pérdida en texto no visto y predictibilidad n-grama | Ordenan las variantes **al revés** en toda semilla y tamaño (Spearman 0.98–1.00 entre pérdida y razón) |

El modelo generaliza tarde. La razón del modelo pequeño con 16 rellenos es solo 0.17–0.19, y cruza
0.5 con una cobertura de 0.67–0.80. El modelo más grande necesita más, y llega a 0.554 contra 0.901
con 24 rellenos, de modo que "suficiente" depende del tamaño del modelo, punto al que volvemos en
§6. Los pares mínimos del tipo "cut the tomato / cut the slowly", pensados como control fácil,
siguieron a la razón (0.25 con un relleno) en vez de quedarse cerca del 100%. La Figura 2 muestra ambas lecturas.

![**Figura 2.** Gramática de recetas. (a) Cobertura de la posición de ingredientes de cada verbo frente a la razón de generalización, en tres semillas y dos tamaños de modelo (puntos: verbos; líneas: medias por variante). La cobertura predice qué verbos generalizan (Spearman 0.94–0.98), y el modelo más grande necesita más cobertura. (b) Entre variantes del corpus (etiquetas: ingredientes por verbo), una menor pérdida en texto no visto va con una menor generalización: los corpus más restringidos son los más predecibles.](figures/fig2_recipe.png){width=100%}


### 5.2 Nivel de cabeza en texto natural: huecos plantados y dosis

*Detección, con la herramienta final.* Plantamos cinco cabezas de una muestra de FineWeb (*people,
she, But, only, even*) en una parte y diagnosticamos cada parte contra la muestra.

| Plantado | Cabezas encontradas (de 5) | Otras filas marcadas |
|---|---|---|
| oraciones quitadas, k = 2 | 4 | 0 |
| rellenos reemplazados, k = 2, 4, 8, 16 | 5 en cada k | 0 |
| rellenos reemplazados, k = 32 | 4 | 0 |
| 7 partes sin plantar | 0 | 0 |

Toda cabeza encontrada se encuentra por su fila de rellenos prenominales (la clase de *the*, *a*, los
posesivos y los adjetivos, §5.8), con diversidad
0.03–0.17 para k = 2 y 0.27–0.35 para k = 32, frente a alrededor de 0.9 en las partes sin plantar.
La mayoría también se lee *ausente* en una segunda clase de relleno. Ninguna fila fuera de las
cabezas plantadas se marca en ninguna parte, de 2,900–6,600 filas leídas. El nivel de clase marca
una fila de clase en cada una de dos de las siete partes sin plantar (§7).

*Costo en el modelo.* Los modelos entrenados con los corpus plantados dan a la palabra siguiente
verdadera 2.17–4.60 nats menos (k = 2) y 1.12–3.60 nats menos (k = 16) en toda cabeza plantada y
semilla, mientras las cabezas de comparación quedan dentro de 0.3 nats en 19 de 20 lecturas. La
pérdida sobre todo el texto se mueve solo 0.03 bits por byte: registra el daño, pero no dice dónde.

*Dosis.* Plantamos catorce cabezas con diversidad 0.1–0.7, en dos asignaciones y dos semillas. La
pérdida es monótona en la dosis (Spearman 0.76–0.92), y las cabezas en 0.1 pierden 2.8–3.3 nats. El
extremo suave no sale gratis: las cabezas con diversidad 0.5–0.7 pierden 0.38–0.57 nats; por lo
tanto, la diversidad no es el tamaño del efecto. D es el tamaño del efecto (caída ≈ −0.72 · D,
R² 0.78), aunque en los plantados no ordena las cabezas mejor que la diversidad. La Figura 3 muestra cada cabeza plantada.

![**Figura 3.** Dosis y respuesta en FineWeb. Catorce cabezas plantadas con diversidad 0.1–0.7 (dos asignaciones, dos semillas cada una), frente a la caída de la log-probabilidad que el modelo da a la palabra siguiente verdadera tras la cabeza, respecto de un modelo entrenado con el corpus sin plantar. La pérdida es monótona en la dosis (Spearman 0.76–0.92) y no desaparece en el extremo suave. Las cabezas de comparación sin plantar quedan en su mayoría cerca de cero.](figures/fig3_dose.png){width=60%}


### 5.3 Recortes realistas de un corpus

Cinco recortes del 10–50% de F se diagnosticaron contra F, y se entrenaron seis modelos por
semilla (§4.2).

| Recorte | Tokens/palabra | Insuficientes | Ausentes | Filas de clase marcadas |
|---|---|---|---|---|
| aleatorio 50% | 1.49 | 0 | 0 | 0 |
| aleatorio 25% | 1.49 | 0 | 0 | 0 |
| aleatorio 10% | 1.52 | 0 | 0 | 0 |
| 25% menos parecido a Wikipedia | 1.39 | 2 | 11 | 4 |
| 25% más formulaico (menor perplejidad) | 1.61 | 1 | 4 | 1 |

*Los recortes aleatorios salen aleatorios.* No marcan nada, ni filas ni clases.

*Los filtros son los únicos recortes marcados, pero levemente.* El filtro tipo Wikipedia marca ","
y ":" antes de la clase prenominal y de etiquetas, y el filtro formulaico marca ". Our". Sus filas ausentes
son unas pocas cabezas pobres ("50", " a day", " they should"). Habíamos predicho al menos 10 filas
insuficientes para cada filtro, y ambas predicciones fallaron.

*El z de una cabeza predice lo que el recorte le cuesta al modelo mejor que su frecuencia.* En el
recorte del 10%, Spearman(z, caída) sobre 416 cabezas es 0.157 / 0.181 (dos semillas), y el del log
del tronco solo, 0.037 / 0.018.

*Los filtros mueven las posiciones.* El puntaje de un recorte es el exceso de sus cabezas,
E = D − media de D en 8 recortes aleatorios de F de su tamaño, ponderado por las proporciones de
rellenos de F, y z lo divide por la dispersión de los recortes dejando uno fuera. El filtro de
perplejidad que conserva el 25% de documentos más formulaicos da z = +18.6, el que conserva los
documentos menos parecidos a Wikipedia da z = +24.8, y el filtro tipo CCNet con Wikipedia da
z = +9.7. Los recortes aleatorios dan |z| ≤ 1.5.

*El modelo paga donde apunta el exceso.* Dentro de un recorte filtrado, las cabezas con mayor E
pierden más de lo que cuesta un recorte aleatorio del mismo tamaño: Spearman(E, caída − caída
aleatoria) va de −0.24 a −0.37, en seis filtros y ambas semillas. El aparente éxito de D crudo en
recortes aleatorios era solo tamaño de muestra. En el recorte del 10%, D da −0.42 / −0.45 frente a
la caída, y E solo −0.05 / −0.11.

*Elegir entre recortes.* Sobre diez recortes del 25%, el puntaje del corpus ordena solo en parte
los bits por byte de los modelos: Spearman 0.65 / 0.55, frente a una predicción de al menos 0.5 en
ambas semillas. El número de documentos, un indicador indirecto de la cobertura temática, pesa lo
mismo (−0.47 / −0.44), y el puntaje ordena lo que deja el número de documentos con 0.58 / 0.56. A
lo largo de la dosis de un mismo filtro, con los documentos fijos, el puntaje ordena los modelos
(Spearman 0.90); cuando la dosis también suma documentos, no (0.60).

### 5.4 La sintaxis se aprende a nivel de clase

*Primera lectura, negativa.* Entre recortes realistas, G, leída entonces contra cualquier clase
abierta y antes de los contrastes de §4.3, siguió al tamaño (F 0.748, recortes del
25% 0.700–0.736, del 10% 0.69) y al número de documentos, no al puntaje (−0.16 / −0.33). G por
cabeza se reproduce entre semillas (Pearson 0.87–0.91). La razón, que encontramos después, es que
los filtros realistas empobrecen la posición de una cabeza, pero nunca vacían el patrón de su clase.

*Eliminar en una cabeza frente a eliminar en la clase.* A partir de r50 y con el mismo tamaño en
bytes, construimos tres corpus. El corte de clase quita toda oración en que un pronombre (*I you he
she we they*) o un modal va seguido de un verbo; el corte de cabeza quita solo las de *she, they,
can, will*; y el control quita oraciones al azar.

| Clase | ΔG corte de clase | Cabezas quitadas: corte de cabeza / corte de clase | Hermanos en el corte de cabeza |
|---|---|---|---|
| pronombres | −0.195 | −0.044 / −0.177 (25%) | −0.028 |
| modales (ítems VB) | −0.126 | −0.040 / −0.081 (49%) | −0.008 |
| posesivos | −0.303 | −0.194 / −0.343 (57%) | +0.046 |

Quitar el patrón de la clase hace perder la sintaxis a todos sus miembros. Quitárselo a dos cabezas
les cuesta entre un cuarto y algo más de la mitad, y deja intactos a sus hermanos.

El nivel de clase del indicador coincide con los modelos. En el corte de clase la herramienta
reporta el nodo de los modales (t 319) y el nodo de pronombres *I you we they* (t 254), frente a un
umbral de clase de 13.9. En el corte de cabeza no reporta ningún nodo de pronombres, y sí un nodo de
cinco modales que contiene *can* y *will* con t 33, un décimo del del corte de clase. D por cabeza,
en cambio, dice que *she* y *they* perdieron casi lo mismo en ambos cortes (*she* +1.60 / +1.73,
*they* +1.28 / +1.39, corte de cabeza / corte de clase, frente al control).

Por lo tanto, la sintaxis se aprende en ambos niveles, sobre todo en el de clase. Cuánto suma la
evidencia propia de una cabeza depende de la clase: lo mínimo en pronombres, lo máximo en
posesivos.

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

En los cortes marcados, ninguna configuración recupera la clase (0.13–0.27 por debajo de F). Más
capacidad lo empeora: los pronombres van 0.85 → 0.75 → 0.68 y los modales 0.87 → 0.80 → 0.75, ya
que un modelo más grande memoriza con más fuerza la ausencia del patrón. En los controles sin
marca, el mejor control queda a menos de 0.02 de F. En un recorte sin marcas más chico (10%),
cambiar de modelo movió G hasta 0.06; esa es la palanca del ingeniero.

*Con cuatro veces más datos.* BigF (57.8 MB frente a los 14.4 MB de F), su corte de clase y su control se entrenaron con el modelo
mediano durante 30,000 pasos.

| Modelo | BLiMP | G pronombres | G modales |
|---|---|---|---|
| BigF | 0.629 | 0.910 | 0.902 |
| control | 0.632 | 0.822 | 0.895 |
| corte de clase | 0.622 | 0.761 | 0.565 |

BLiMP sube de 0.57 a 0.63. El control sin marcas ya no alcanza la sintaxis de F con una
diferencia de 0.02: sus pronombres quedan 0.09 por debajo de los de BigF. En los modales, el resultado de atribución es más fuerte a escala: el
corte deja a los modales casi en el azar. En los pronombres, la brecha baja a 0.06, y nuestras
predicciones fallaron. Con este tamaño de datos, casi todo verbo en pasado frecuente sigue a cada
pronombre en la referencia, lo que deja 3–7 ítems no vistos por pronombre, y las lecturas por
cabeza se vuelven erráticas. No podemos separar la falta de potencia de la prueba de una reducción
real. Más datos también traen más evidencia indirecta (sujetos sustantivos), como reporta FiCT. La Figura 4 reúne los resultados de nivel de clase de §5.4 y §5.5.

![**Figura 4.** ¿Corpus o modelo? (a) G de los modelos entrenados con el conjunto completo F, con los controles sin marca del mismo tamaño, con los cortes de clase marcados y con un recorte del 10% sin marca, en cada configuración de modelo; "scale" es el modelo mediano entrenado con cuatro veces más datos (BigF y sus cortes). Ninguna configuración recupera una clase marcada, mientras que los controles sin marca quedan a menos de 0.02 de F a escala pequeña (a escala, los pronombres del control quedan 0.09 por debajo de los de BigF). (b) Cambio de G respecto del control: quitarle un patrón a toda la clase le cuesta a cada miembro, mientras que quitárselo a dos cabezas les cuesta entre el 25 y el 57% de eso y deja intactos a sus hermanos.](figures/fig4_attribution.png){width=100%}


### 5.6 La detección de la herramienta

Diagnosticamos los 22 cortes etiquetados de §4.1 contra F con 1.50 tokens por palabra. Las filas de
clase encuentran cada clase eliminada, cada una en su propia rama del árbol, y no marcan ningún
control.

| Corte | Nodo reportado | t (umbral de clase 13.9) |
|---|---|---|
| corte de clase de pronombres y modales | 8 cabezas, entre ellas *can will would may could should* | 319 |
| | *I you we they i* | 254 |
| corte de clase de posesivos | *your his my their* | 165 |
| corte de clase de números | *two … seven* | 104 |
| corte de clase de verbos base | *get make see take find leave* | 101 |
| corte de cabeza de pronombres y modales | *can will would could should* | 33 |
| corte de cabeza de posesivos | *his their*, exactamente las cabezas quitadas | 85 |
| cortes de cabeza de números y de verbos base | nada | |
| 12 controles aleatorios | nada | |

Las filas de cabeza separan los cortes de los controles de la misma manera: el corte de clase y el
corte de cabeza marcan 39 y 13 filas de cabeza, y los 12 controles, ninguna. Las filas *absent*
encuentran cada una de las 8 palabras borradas. En 8 controles aleatorios nuevos, dos filas se leen
ausentes, una en cada uno de dos controles: ` luck`, mostrada 0 veces frente a 7.0 ± 1.8 en los
muestras, y `Maybe`.

Por último, leímos 48 cortes aleatorios de documentos enteros de F, que son muestras aleatorias de
la referencia, contra un mismo conjunto fijo de muestras. Dan 2 marcas de cabeza (4%), ninguna fila
de clase y ningún *absent*. alpha es 0.06 por familia.

### 5.7 Comparación con métricas de corpus y con BLiMP

*Métricas de un solo número.* Sobre los cortes de clase, los cortes de cabeza y ocho controles
aleatorios calculamos la perplejidad bajo el bigrama de F y bajo un bigrama de Wikipedia, la KL de
unigramas desde F, MATTR (Covington & McFall, 2010), distinct-2 (Li et al., 2016) y el puntaje
Vendi (Friedman & Dieng, 2023) de embeddings de oraciones MiniLM (Wang et al., 2020; Reimers &
Gurevych, 2019). La prueba planificada (3 desviaciones estándar de los controles) marca como mucho
un corte de clase por métrica. Esa prueba fue indulgente: los controles mezclaban tamaños, y la
perplejidad media tenía un valor atípico. En una lectura a igual tamaño, que tratamos como
exploratoria, la KL de unigramas y distinct-2 sí registran los dos cortes de clase grandes, que
quitaron 18,126 y 10,315 oraciones (KL +0.019 y +0.011, frente a un ruido de alrededor de 0.001).
Sin embargo, se quedan dentro del ruido en los números (1,972 oraciones) y en los verbos base
(3,642), que el indicador marca con t 104 y 101, y ninguna métrica dice qué patrón cambió. Lo que
estos resultados respaldan es, por lo tanto, la localización y la sensibilidad a pérdidas pequeñas
y específicas de un patrón, no que las otras métricas no vean nada.

*BLiMP.* Nuestros modelos pequeño y mediano sacan 0.56–0.58 en BLiMP (azar 0.5), y 0.63 a escala;
la concordancia sujeto–verbo queda alrededor de 0.52 a escala pequeña. Entre control y corte de
clase, BLiMP se mueve como mucho 0.017, mientras que G se mueve 0.11–0.34: un solo puntaje sobre 67
paradigmas diluye una pérdida específica de una clase. G y BLiMP, además, miden sintaxis distinta.
G lee las clases de las posiciones locales, que nuestros modelos aprenden bien (0.96–1.00 en
pronombres y modales), mientras que los paradigmas de BLiMP dependen sobre todo de dependencias más
largas.

### 5.8 Sin ajustes arbitrarios

Cada umbral fijado a mano de la versión anterior (§4.4) se derivó de los datos o se midió contra
los veredictos sobre los 22 cortes etiquetados de §5.6.

*Granularidad.* Se aprendió un tokenizador sobre F hasta 60,000 fusiones y se cortó en cada
granularidad.

| Tokens/palabra | Fusiones | Clases encontradas (de 5) | Controles con marca de clase (de 12) | Controles con marca de cabeza (de 12) | Palabras borradas |
|---|---|---|---|---|---|
| 1.35 | 30,526 | 5 | 1 | 0 | 8/8 |
| 1.40 | 20,608 | 5 | 0 | 0 | 8/8 |
| 1.50 | 11,381 | 5 | 0 | 0 | 8/8 |
| 1.60 | 7,220 | 5 | 1 | 1 | 8/8 |
| 1.70 | 4,944 | 5 | 0 | 3 | 8/8 |
| 1.80 | 3,548 | 5 | 0 | 0 | 8/8 |

Los veredictos se mantienen en todo el rango, y las marcas en controles (6 de 144 lecturas) están
cerca de la tasa de diseño. Entre granularidades vecinas, insuficiente y dentro del ruido se
intercambian en 26–37 de 330,000–690,000 patrones; lo que se mueve es si un patrón raro se lee o
no. Una granularidad de 1.30 no entra en 16 GB, ya que deja 45,748 tokens para agrupar. Solo en
F, el rango no tiene un borde medido en ninguno de los dos lados; sus bordes vienen de los otros
dominios, más abajo. Un presupuesto fijo, en cambio, no significa nada entre
corpus: 4,000 fusiones dejan 1.76 tokens por palabra en F, mientras que en 30 documentos de recetas
las fusiones se agotan en 269, cerca de palabras enteras. Por eso la herramienta aprende hasta una
granularidad, no hasta un presupuesto, y rechaza un presupuesto fuera del rango.

*Dominios, tamaño y conjunto.* Repetimos la lectura en 11 granularidades, de 1.30 a 2.50, en cinco
corpus: TinyStories, FineWeb, Wikipedia y dos dominios de conocimiento, resúmenes de PubMed y los
hechos de casos del Tribunal Europeo de Derechos Humanos. Cada corpus se leyó a 0.9, 3.6, 14.4 y
57.6 MB, con tres conjuntos disjuntos en cada tamaño hasta 14.4 MB y uno a 57.6 MB, cada conjunto
con sus propios 22 cortes etiquetados: 550 lecturas en total. Una granularidad se sostiene cuando
se encuentra cada clase eliminada y cada palabra borrada que es un solo token se lee ausente.

| Tamaño | Conjuntos donde 1.50 se sostiene | Rango común a todos los corpus |
|---|---|---|
| 0.9 MB | 1 de 15 | ninguno |
| 3.6 MB | 6 de 15 | ninguno |
| 14.4 MB | 12 de 15 (14 de 15¹) | 1.35–1.70 |
| 57.6 MB | 5 de 5 | 1.35–2.00 |

¹ Dejando fuera una palabra borrada que ninguna granularidad del conjunto lee ausente (ver abajo).

Desde 14.4 MB, los veredictos se sostienen de 1.35 a 1.70 en los cinco dominios, en todos los
conjuntos menos uno al dejar fuera las palabras borradas que no se pueden resolver¹; el tercer
conjunto de FineWeb falla por una palabra borrada (más abajo). 1.30 falla en el segundo conjunto de
FineWeb y las fusiones de PubMed se agotan antes, y 1.80 falla en dos de los tres conjuntos del
corpus legal. El
borde superior se mueve con el conjunto y crece con el tamaño: a 57.6 MB, Wikipedia se sostiene
hasta 2.50 y el corpus legal hasta 2.20. Las falsas alarmas de los controles quedan bajo el doble
del nivel de la prueba en los 50 corpus. El rango de la herramienta es entonces 1.35–1.70, y 1.50
sigue dentro.

Bajo 14.4 MB las lecturas son erráticas, y la referencia es el mismo conjunto pequeño. Las clases
eliminadas no se encuentran en 86 de las 134 lecturas que fallan a 0.9 MB, sobre todo los verbos
base y los números. La causa no son los ajustes del árbol de clases: el piso de ruido de su corte
nunca actúa, y un contexto igualado en conteo a la medición de 14.4 MB recuperó los posesivos de
TinyStories pero ninguna clase de Wikipedia ni del corpus legal. Por eso la herramienta avisa
cuando su dominio tiene menos de 14.4 MB.

Las fallas a 14.4 MB son palabras borradas sueltas cuyos usos están en pocos documentos:
*assessment* tiene el 59% de sus usos en un documento de FineWeb, *retrieved* el 68% en tres,
*oil* el 80% en un artículo de Wikipedia. Frente a muestras aleatorias de documentos enteros, la
ausencia de una palabra así se confunde con una muestra que no tocó su documento, en toda granularidad
del rango (*retrieved* se lee ausente solo a 1.30). Las ocho palabras elegidas a mano para F están repartidas en el 5% al 15% de los
documentos, y ninguna falló.

*Las clases frente a un etiquetador.* Comparamos el corte y el árbol con las etiquetas
universales de spaCy (`en_core_web_sm`, nunca el indicador) en un conjunto de 14.4 MB de cada
dominio a 1.50. Un token que es una palabra entera toma la etiqueta que reúne el 90% de sus 20 o
más apariciones; estos tokens son el 39–46% de los tokens del texto, y el resto son pedazos de
palabra, puntuación, números y palabras ambiguas. Frente a las etiquetas barajadas sobre los mismos
tokens, la pureza del corte es 0.67–0.82 (las barajadas, como mucho 0.51) y su índice de Rand
ajustado 0.39–0.69 (como mucho 0.03). Los sustantivos y los verbos tienen cada uno un nodo del
árbol con F1 0.76–0.92 en los cinco dominios, y los adjetivos con 0.64–0.81. Las clases cerradas
no forman clases propias: *the*, *a*, los posesivos y los adjetivos comparten la clase de lo que va
antes de un sustantivo en FineWeb, PubMed y el corpus legal, y *of*, *and*, *to* comparten la de
los verbos en FineWeb, Wikipedia y PubMed. Toda etiqueta leída supera a todas las barajadas en
todos los dominios. El corte elegido queda entre el 2.º y el 11.º de los 57 cortes de 8 a 64
clases por índice de Rand ajustado. Las clases son, por lo tanto, posiciones distribucionales, y
una eliminación limitada a una clase cerrada (solo los posesivos) se lee en un nodo dentro de su
clase; por eso las filas de clase se leen en cada nodo.

*Los ajustes de clase.* Forzamos el corte grueso de 8 a 64 clases, y el contexto de 250 a 4,000
tokens (del 51% al 87% de los tokens del texto). En cada valor se encuentran las cinco clases, no
se marca la fila de clase de ningún control, las marcas de cabeza en controles quedan en 0–2 de 12,
y se encuentran todas las palabras borradas. Las filas de clase no dependen en absoluto del corte
grueso: se leen sobre todo el árbol, y el umbral de clase es 13.92 en todos los casos. Una fila de
cabeza, en cambio, es una cabeza seguida de una clase de relleno, así que otro corte le hace otra
pregunta a la misma cabeza; la coincidencia a nivel de patrón con el corte de 12 clases baja del
97% (8 clases) al 63% (64).

*El umbral y las muestras.* Con 16 muestras y el máximo como umbral, se marcaron 6 de 48 cortes
aleatorios contra un conjunto de muestras, y ninguno con muestras nuevas para cada corte. El umbral
estaba calibrado en promedio sobre conjuntos de muestras, pero el máximo de 16 muestras también
tiene ruido (de 5.6 a 9.6 entre conjuntos). Una reserva de 200 muestras coincidió con la ley Beta de
§3.4 con 16–128 muestras. Con 101 muestras y k = 6, se marcan 2 de 48.

*Lo que reemplazaron.* El piso de diversidad de 0.5 y el margen de 3 sigma habrían necesitado una
recalibración para cada corpus nuevo. Cuando todo token pasó a ser relleno, el piso de clase de
0.5 nats no detectó ninguna clase eliminada salvo una, y ningún piso separaba los desplazamientos
de clase que les costaron la sintaxis a los modelos de los que no. El tronco mínimo de 20, barrido
de 2 a 40, no cambió ningún veredicto entre 5 y 40 en F, pero movió ambos umbrales, y no se
trasladaría a un corpus de otro tamaño.

### 5.9 Sin referencia

La saturación sola no marca una posición empobrecida. Empobrecimos el patrón pronombre → verbo al
5–50% de sus oraciones con los mismos bytes y leímos cada corpus por sí solo. La masa faltante de
la posición sube de 0.028 a alrededor de 0.05, pero el nodo mediano de cada corpus da 0.40, así que
cualquier tolerancia que marcara la posición empobrecida marcaría la mayor parte del corpus. Si se
quita por completo, la posición vuelve a dar 0.028, tan saturada como el control.

Comparar una cabeza con sus hermanos sí marca una posición empobrecida. Leída con la prueba de
hermanos de §3.5, *they* en el corte de cabeza de §5.4 muestra 3 verbos donde una partición
aleatoria de su padre da 26.5 (t 9.6, umbral 6.3). La herramienta marca *she* y *they*, y ningún
pronombre del control. Las marcas se replican: las dos mitades aleatorias del control, leídas sobre
el árbol y las clases del propio control para que cada prueba sea la misma prueba, comparten 264
pruebas débiles, de 332 y 397. En la gramática de recetas, 11 de 14 verbos restringidos salen
débiles; los otros tres comparten su único ingrediente con su hermano, y ningún verbo sale débil en
las recetas diversas. En FineWeb la herramienta marca 95 filas de cabeza en el control y 120 en el
corte de cabeza, en 2.5 minutos.

La mayoría de estas marcas son cabezas que la gramática restringe dentro de su clase de relleno:
*to* toma menos rellenos que las otras preposiciones (524 frente a 871), y los verbos de reporte
(*said, says, means*), menos que otros verbos. Son débiles en el corpus, que es lo que ve un modelo
entrenado con él. Una referencia las lee dentro del ruido, porque el dominio también las tiene.

La prueba no puede ver algunos casos. Empobrecer todos los pronombres a la vez no deja ningún
hermano más fuerte: el nodo de pronombres da 0.94–1.06 de sus rellenos esperados cuando se
empobrece al 25% o al 10%, y el corte de clase muestra su posición 19 veces. En un corpus sintético
de sujeto–verbo, *she* y *they* restringidos a dos verbos forman su propia clase y pierden a su
hermano.

Descartamos tres alternativas. Comparar cada cabeza con toda su clase gruesa, agregar una prueba de
con qué frecuencia una cabeza toma cada clase de relleno, y agrupar cabezas por contexto marcaron
500–2,000 cabezas y 3,000–15,000 filas de clase en el control, entre ellas todos los pronombres.
Sus mitades compartieron el 12–17% de las marcas.

---

## 6. Discusión

*Qué certifica el indicador.* Un patrón de clase marcado es una falla del corpus. En nuestros
experimentos ninguna configuración lo recuperó, y más capacidad lo empeoró. El remedio son datos:
texto en que las cabezas de la clase tomen otros rellenos de su posición. A la inversa, un corpus
sin marcas permite alcanzar la sintaxis de la referencia para su tamaño, y ahí una falla apunta al
modelo. Esta división del trabajo es el punto práctico del trabajo. El lado de los datos certifica
que los patrones están servidos; encontrar el modelo que los aprenda queda para la ingeniería del
modelo.

*Por qué a nivel de clase.* Los modelos comparten evidencia entre las cabezas de una clase: una
cabeza con una posición pobre la toma prestada de sus hermanos, lo que coincide con los resultados
de evidencia indirecta de FiCT y del estudio AANN. Los filtros realistas empobrecen cabezas, pero
rara vez vacían clases, lo que explica por qué, entre recortes realistas, la sintaxis a nivel de
clase siguió al tamaño y al tema y no al puntaje (§5.4). Un diagnóstico pensado para la sintaxis
debe, por lo tanto, leer clases, mientras que un diagnóstico de qué palabra predecirá un modelo lee
cabezas. Leer cada nodo del árbol de cabezas, en vez de un solo corte, permite que el reporte
nombre el grupo que carga un desplazamiento: toda la clase de los modales en un corte, dos
posesivos en otro.

*Ajustes a partir de los datos.* Un umbral fijado sobre los corpus en que se desarrolló un método
es una afirmación sobre esos corpus. El umbral del indicador, en cambio, dice para cada corpus
cuánto se aleja, en el peor caso, una muestra aleatoria de la referencia. Al lector le queda un
solo número por juzgar, alpha, que es una tolerancia y no una propiedad de los datos. El costo es
tiempo: con 101 muestras, un diagnóstico tarda 4–9 minutos con nuestros tamaños, frente a unos
4 minutos antes.

*Con y sin referencia.* Sin referencia, el corpus se juzga por su propia evidencia: una cabeza más
débil que sus hermanos es débil para un modelo entrenado con él. Esta es la lectura que tiene quien
trabaja con estos modelos antes de encontrar una referencia, y marca huecos reales (*she* y *they*
en el corte de cabeza). Una referencia hace dos cosas más. Ve lo que al corpus le falta por
completo, o le falta a toda una clase, y separa la debilidad del corpus de la del dominio. Ambas
lecturas son relativas. "Sin marcas" significa tan bueno como una muestra aleatoria de la
referencia de este tamaño, o como una parte aleatoria del padre de la cabeza; no significa tan
bueno como la referencia a cualquier tamaño. En nuestros datos, un recorte aleatorio de un corpus
grande y bien construido ya conserva los patrones tan bien como cualquier filtro. El papel práctico
del indicador es sobre todo rechazar recortes y filtros que sesgan la sintaxis.

*Tamaños de efecto.* Los efectos del daño a las posiciones sobre el modelo completo son pequeños en
bits por byte (0.017 en toda una escalera de un filtro), mientras que los efectos por patrón son
grandes: 1–4.6 nats en cabezas plantadas y 0.13–0.30 de G en eliminaciones de clase (§5.4). Hace falta un
diagnóstico por patrón precisamente porque los números del corpus completo y del modelo completo
diluyen las pérdidas locales.

---

## 7. Limitaciones

La primera limitación es la escala. Los modelos llegan a 6.2M de parámetros, y los corpus a unos
10M de palabras, en CPU, y en BLiMP los modelos sacan 0.56–0.63. Nuestras afirmaciones tratan, por
lo tanto, de sintaxis local aprendida por modelos pequeños; no está probado que la atribución a
nivel de clase se mantenga en modelos que aprenden sintaxis de largo alcance. El indicador mismo
cubre solo sintaxis local, es decir, cabezas de uno o dos tokens y el relleno siguiente. Las
dependencias de largo alcance (concordancia a distancia, relleno–hueco, ligamiento) quedan fuera de
alcance.

La evidencia del lado de los modelos viene de un idioma y una fuente natural, inglés y FineWeb,
más una gramática sintética. Del lado del corpus, el rango de granularidad se midió en cinco
dominios en inglés, de 0.9 a 57.6 MB, con tres conjuntos por tamaño hasta 14.4 MB y una a 57.6 MB;
otros idiomas y corpus más grandes no se probaron. Bajo 14.4 MB los veredictos no se sostienen,
así que la herramienta necesita un dominio de al menos ese tamaño; los plantados de §5.2 se
leyeron contra una muestra de 6.5 MB, antes de medir este mínimo, y ahí se sostuvieron. Una palabra borrada cuyos usos
están en pocos documentos no se puede leer ausente dentro del rango de granularidad: el indicador resuelve la
ausencia de palabras repartidas en muchos documentos, no de palabras concentradas en pocos. Solo
algunas clases se probaron a fondo: tres
(pronombres, modales, posesivos) se entrenaron con eliminación, y dos más (números, verbos base) se
revisaron del lado del corpus. A escala, el resultado de pronombres no se confirmó, porque su
prueba perdió potencia. La mayoría de los experimentos usó dos semillas; las configuraciones nuevas
de modelo y las corridas de escala usaron una.

El nivel de la prueba, alpha = 0.06, es por familia. Un corpus aleatorio puede marcarse en
cualquiera de tres familias, así que su probabilidad de alguna marca es como mucho del 18%; medida,
fue del 4% (2 de 48). Repartir alpha entre las tres necesitaría unos 300 muestras. En las partes
plantadas, se marcó una fila de clase en cada una de dos de las siete partes sin plantar. El borde
superior del rango de granularidad se mueve con el conjunto y crece con el tamaño (hasta 2.00 a
57.6 MB), así que 1.70 es el borde común a todos los dominios a 14.4 MB, no una propiedad de cada
corpus.

Sin referencia, no se ven un patrón que le falte al corpus, una clase mostrada débilmente en su
conjunto, ni una cabeza que el árbol separe de sus hermanos.

Otros dos límites tienen que ver con lo que mide el indicador. En la gramática de recetas, la
diversidad que necesita un modelo creció con su tamaño. El indicador dice dónde un corpus se aparta
de su referencia; cuánto apartamiento tolera un modelo dado no es parte de él. Por último, el
indicador aborda la sintaxis, no la semántica. Un corpus puede pasar y aun así fallarle a un modelo
por tema o significado: un corpus de tema estrecho perjudicó a los modelos en todos los documentos
sin que se marcara ninguna posición. También las clases son distribucionales más que gramaticales:
frente a un etiquetador, los sustantivos y los verbos salen como clases y los adjetivos como un
nodo del árbol, pero los determinantes, los posesivos y los adjetivos comparten la clase de lo que
va antes de un sustantivo, y *of*, *and*, *to* comparten la de los verbos (§5.8); los pedazos de
palabra, la puntuación, los números y las palabras ambiguas, el 54–61% de los tokens del texto, no
tienen etiqueta con la cual compararse.

---

## 8. Conclusión

Preguntamos si un corpus puede revisarse, antes de entrenar, en busca de los patrones sintácticos
locales que su dominio necesita, de modo que una falla de generalización se pueda atribuir al
corpus o al modelo. La respuesta, dentro de los límites anteriores, es sí.

Un indicador que compara los rellenos de cada posición con muestras aleatorias de una referencia,
en cada nodo de un árbol inducido de cabezas, marca fallas del corpus que ninguna configuración de
modelo probada recupera. Donde no marca nada, los modelos pequeños alcanzan la sintaxis de la
referencia. Ninguno de sus ajustes se ajusta a mano: el tronco de lectura y el número de muestras
salen de los datos, la granularidad está dentro de un rango medido, y el nivel de la prueba se declara en cada reporte. Sin referencia, el
mismo árbol permite juzgar un corpus por su propia evidencia. En el camino encontramos que los
modelos aprenden la sintaxis local sobre todo a nivel de clase, lo que explica tanto por qué los
filtros de datos realistas rara vez rompen la sintaxis como por qué un diagnóstico debe leer
clases. La herramienta, el código y cada predicción prerregistrada, incluidas las fallidas, se
publicarán junto con el artículo.

El trabajo futuro incluye modelos y corpus más grandes, con GPU; dependencias de largo alcance;
más idiomas, para probar que el rango medido en cinco dominios en inglés se traslada; la tolerancia de un
tamaño de modelo dado a un apartamiento marcado; y una contraparte semántica para los dominios de
conocimiento.

---

## Apéndice A. Predicciones que fallaron

Las predicciones se conservan tal como se escribieron en los documentos de prerregistro.

**Indicador y modelos (versión anterior del indicador).**

- **Gramática de recetas.** Los pares mínimos "cut the slowly" no se quedaron cerca del 100% (0.25
  con un relleno). Las predicciones de tamaño de corpus fallaron con 500 documentos.
- **Plantados.** Una cabeza de comparación se movió −0.51 nats. La pérdida sobre todo el texto
  subió 0.03 bits por byte (se predijo < 0.02). Con k = 32 todavía se detectó a menudo (12/20).
- **Corpus de tema estrecho.** Las cabezas marcadas no perdieron más; la pérdida fue temática y en
  todos los documentos.
- **Selección de subconjuntos.** Las marcas naturales suaves (diversidad 0.26–0.50) no le costaron
  al modelo. En r10, Spearman(z, caída) fue 0.10 / 0.12 (umbral 0.15).
- **Dosis.** El extremo suave costó 0.38–0.57 nats (se predijo ≤ 0.3), y el umbral se cruzó en
  0.64.
- **Divergencia.** En los plantados, D no ordenó las cabezas mejor que la diversidad. Su primer
  corte, κ, falló por un piso de tamaño de muestra.
- **Elección de recortes.** Sobre diez recortes, Spearman(puntaje, bits por byte) fue 0.55 / 0.45
  (se predijo ≥ 0.5 en ambas semillas). Con la lectura actual se cumple (0.65 / 0.55).
- **Escaleras.** La escalera que suma documentos dio Spearman 0.60. La lectura por cabeza falló con
  dosis del 25%.
- **Evaluación sintáctica de recortes realistas.** Fallaron las seis predicciones, lo que llevó a
  la hipótesis de nivel de clase.
- **Patrones de clase.** P2 dio −0.086 (umbral −0.10), con una primera prueba de modales inválida.
  En P4, el ruido entre modelos superó el umbral.
- **Réplica.** El corte de cabeza de posesivos costó el 57% (umbral 50%).
- **Herramienta.** No se detectó la clase de pronombres (+0.47 frente a un piso de 0.5). La regla
  anterior de *absent* dio 79–114 falsas marcas, y un control dio 5 con m/2.
- **BLiMP.** Los modelos de F sacaron menos de 0.60.
- **Escala.** La brecha de pronombres fue 0.06 (umbral 0.10), y el control quedó 0.09 por debajo de
  la referencia.

**Ajustes a partir de los datos.**

- **Granularidad.** Se predijo que granularidades vecinas coincidirían en el 90% de los patrones;
  coincidieron en el 85–94%, y casi toda la diferencia era si un patrón se lee o no. Con un tronco
  de 20, los números y los verbos base no se detectaron con 1.35, aunque se predijo que se
  encontrarían en todo el barrido.
- **Piso de clase.** Se predijo que cada clase cuya eliminación les costó la sintaxis a los modelos
  daría un exceso mayor que todas las que no. La predicción falló: con 64 clases de cabezas sobre
  todo token, pronombres y posesivos compartían una clase con *the* y *a*, y la movieron +0.045 y
  +0.007. Esto llevó a las filas de clase en cada nodo.
- **El umbral de 16 muestras.** Se marcaron 6 de 48 cortes aleatorios contra un conjunto de
  muestras (diseño: alrededor de 1 en 17). Esto llevó a alpha y a las 101 muestras.
- **Sin referencia.** La saturación (masa faltante) no marcó una posición empobrecida, y tres
  agrupaciones de cabezas dieron marcas que las mitades del corpus no compartían.
- **Selección de subconjuntos, repetida.** Se predijo que ambos filtros marcarían al menos 10
  filas; marcaron 2 y 1.
- **Dominios, tamaño y conjunto.** Se predijo que 1.50 se sostendría en todo tamaño y todo conjunto;
  se sostuvo en 1 de 15 conjuntos a 0.9 MB y en 6 de 15 a 3.6 MB, y falló en tres de quince
  conjuntos de 14.4 MB, cada uno por una palabra borrada concentrada en pocos documentos. Se
  predijo que las lecturas de 57.6 MB coincidirían con las de 14.4 MB; en cambio, los rangos de
  Wikipedia y del corpus legal se ampliaron. Se predijo que PubMed necesitaría más fusiones que
  Wikipedia para llegar a 1.50 (15,066 contra 16,022), y que su rango y el del corpus legal
  contendrían 1.30–2.00 (el del legal termina en 1.70). Esto llevó al mínimo de 14.4 MB y al rango
  1.35–1.70.
- **Las clases frente a un etiquetador.** Se predijo que sustantivos, verbos, adjetivos,
  adverbios, preposiciones, determinantes y pronombres tendrían cada uno un nodo del árbol que
  superara a todas las barajadas de las etiquetas en al menos cuatro de cinco dominios. Los
  determinantes tuvieron menos de 20 tokens etiquetados en TinyStories y en el corpus legal y se
  leyeron en tres; toda etiqueta leída superó a las barajadas.

## Apéndice B. Versión anterior del indicador

El indicador se construyó primero con umbrales fijados a mano (§4.4). Sus resultados, reemplazados
en el cuerpo por los de la herramienta final, fueron los siguientes.

- **Huecos plantados (§5.2).** Con un margen de 3 sigma, el barrido de reemplazo encontró 5, 5, 5,
  5 y 4 de 5 cabezas en FineWeb con k = 2, 4, 8, 16 y 32, y 19, 20, 19, 17 y 12 de 20 cabezas en
  cuatro fuentes (FineWeb, Wikipedia, arXiv, C4). Quitando oraciones encontró 3 de 5: dos cabezas
  que quedaron con 6 y 23 apariciones se leyeron apenas por encima de un piso fijo de diversidad
  de 0.5.
- **Recortes realistas (§5.3).** El margen de 3 sigma marcaba 2–8 filas en cada recorte aleatorio,
  a la tasa que le daba el azar. En el recorte del 10%, Spearman(z, caída) era 0.10 / 0.12, con el
  log del tronco solo en 0.10 / 0.11. Leyendo solo tokens de tipo palabra, los puntajes de los
  recortes eran z = +15.0 (filtro formulaico), +20.6 (menos parecido a Wikipedia) y +3.9 (tipo
  CCNet), y |z| < 0.6 en los recortes aleatorios. Spearman(E, caída − caída aleatoria) iba de
  −0.21 a −0.38, y Spearman(puntaje, bits por byte) sobre diez recortes era 0.55 / 0.45.
- **Detección de la herramienta (§5.6).** Con 64 clases de cabezas y un piso de clase de 0.5 nats,
  las filas de clase encontraron 4 de 5 clases eliminadas, y se les escaparon los pronombres, con
  +0.47 nats. Los posesivos se encontraron solo a través de sus cabezas en falta, porque compartían
  clase inducida con *the* y *a*. La primera regla de *absent* se disparó 79–114 veces en
  controles aleatorios.
- **Ajuste de dosis y D por cabeza (§3.3, §5.2, §5.4).** Con la lectura de tokens de tipo
  palabra, el ajuste de dosis era caída ≈ −0.68 · D (R² 0.77), y la D por cabeza daba a *she*
  +1.46 / +1.58 en los cortes de cabeza y de clase.

## Apéndice C. Reproducibilidad

Cada experimento tiene su propio script, sus resultados (JSON) y un registro de su diseño,
predicciones y resultado, escrito antes de correrlo. El diagnóstico está empaquetado como
herramienta de línea de comandos (`corpus-diagnose`) con una interfaz web. Todo corre solo en CPU;
un diagnóstico tarda 4–9 minutos con nuestros tamaños, después de leer la referencia una vez. El
código, los datos y los registros de prerregistro están disponibles a pedido del autor y se
publicarán junto con el artículo.

---

## Declaración de uso de IA generativa

Durante este trabajo el autor usó Claude (Anthropic) como apoyo para escribir código, correr y
analizar experimentos, y redactar el manuscrito. El autor dirigió el estudio, revisó y editó todo
el contenido, y asume plena responsabilidad por él.

---

## Referencias

- Albalak, A., Elazar, Y., Xie, S. M., Longpre, S., Lambert, N., Wang, X., et al. (2024). A Survey on Data Selection for Language Models. *Transactions on Machine Learning Research*. arXiv:2402.16827.
- Baayen, R. H. (2009). Corpus linguistics in morphology: Morphological productivity. In A. Lüdeling & M. Kytö (Eds.), *Corpus Linguistics: An International Handbook* (Vol. 2, pp. 899–919). Berlin: Mouton de Gruyter. https://doi.org/10.1515/9783110213881.2.899
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian Knot: The Moving-Average Type–Token Ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94–100. https://doi.org/10.1080/09296171003643098
- Friedman, D., & Dieng, A. B. (2023). The Vendi Score: A Diversity Evaluation Metric for Machine Learning. *Transactions on Machine Learning Research*. arXiv:2210.02410.
- Good, I. J. (1953). The Population Frequencies of Species and the Estimation of Population Parameters. *Biometrika*, 40(3–4), 237–264. https://doi.org/10.1093/biomet/40.3-4.237
- Guo, Y., Shang, G., & Clavel, C. (2025). Benchmarking Linguistic Diversity of Large Language Models. *Transactions of the Association for Computational Linguistics*, 13, 1507–1526. https://doi.org/10.1162/tacl.a.47
- Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. Zenodo. https://doi.org/10.5281/zenodo.1212303
- Hu, J., Gauthier, J., Qian, P., Wilcox, E., & Levy, R. (2020). A Systematic Assessment of Syntactic Generalization in Neural Language Models. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 1725–1744). https://doi.org/10.18653/v1/2020.acl-main.158
- Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models. In *Proceedings of NAACL-HLT 2016* (pp. 110–119). https://doi.org/10.18653/v1/N16-1014
- Linzen, T., Dupoux, E., & Goldberg, Y. (2016). Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies. *Transactions of the Association for Computational Linguistics*, 4, 521–535. https://doi.org/10.1162/tacl_a_00115
- Luo, J., & Steinert-Threlkeld, S. (2026). Exposure is Optional: Learning Unlike Coordination in Language Models. arXiv:2607.20251.
- Marvin, R., & Linzen, T. (2018). Targeted Syntactic Evaluation of Language Models. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 1192–1202). https://doi.org/10.18653/v1/D18-1151
- Miranda, B., Lee, A., Sundar, S., Casasola, A., Schaeffer, R., Obbad, E., & Koyejo, S. (2023). Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data. arXiv:2306.13840.
- Misra, K., & Mahowald, K. (2024). Language Models Learn Rare Phenomena from Less Rare Phenomena: The Case of the Missing AANNs. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 913–929). https://doi.org/10.18653/v1/2024.emnlp-main.53
- Patil, A., Jumelet, J., Chiu, Y. Y., Lapastora, A., Shen, P., Wang, L., Willrich, C., & Steinert-Threlkeld, S. (2024). Filtered Corpus Training (FiCT) Shows that Language Models Can Generalize from Indirect Evidence. *Transactions of the Association for Computational Linguistics*, 12, 1597–1615. https://doi.org/10.1162/tacl_a_00720
- Penedo, G., Kydlíček, H., Ben Allal, L., Lozhkov, A., Mitchell, M., Raffel, C., et al. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. In *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track* (pp. 30811–30849). arXiv:2406.17557.
- Qi, X., He, L., Roth, D., & Fu, X. (2026). Demystifying Supervision Data Generalization in Multimodal LMs. In *International Conference on Learning Representations (ICLR 2026)*. arXiv:2603.19688.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI technical report.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982–3992). https://doi.org/10.18653/v1/D19-1410
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1715–1725). https://doi.org/10.18653/v1/P16-1162
- Tarone, R. E. (1990). A Modified Bonferroni Method for Discrete Data. *Biometrics*, 46(2), 515–522. https://doi.org/10.2307/2531456
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. In *Advances in Neural Information Processing Systems 33* (pp. 5776–5788). arXiv:2002.10957.
- Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective Function. *Journal of the American Statistical Association*, 58(301), 236–244. https://doi.org/10.1080/01621459.1963.10500845
- Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.-F., & Bowman, S. R. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *Transactions of the Association for Computational Linguistics*, 8, 377–392. https://doi.org/10.1162/tacl_a_00321
- Wei, J., Garrette, D., Linzen, T., & Pavlick, E. (2021). Frequency Effects on Syntactic Rule Learning in Transformers. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 932–948). https://doi.org/10.18653/v1/2021.emnlp-main.72
- Wenzek, G., Lachaux, M.-A., Conneau, A., Chaudhary, V., Guzmán, F., Joulin, A., & Grave, E. (2020). CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data. In *Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020)* (pp. 4003–4012).
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment*. New York: Wiley.
- Wettig, A., Gupta, A., Malik, S., & Chen, D. (2024). QuRating: Selecting High-Quality Data for Training Language Models. In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 52915–52971.
- Witten, I. H., & Bell, T. C. (1991). The Zero-Frequency Problem: Estimating the Probabilities of Novel Events in Adaptive Text Compression. *IEEE Transactions on Information Theory*, 37(4), 1085–1094. https://doi.org/10.1109/18.87000
- Xie, S. M., Santurkar, S., Ma, T., & Liang, P. (2023). Data Selection for Language Models via Importance Resampling. In *Advances in Neural Information Processing Systems 36* (pp. 34201–34227). arXiv:2302.03169.
