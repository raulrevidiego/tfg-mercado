# Resultados

> Borrador del capítulo de resultados. Cifras a 24 de noviembre de 2026, obtenidas con el código y los datos del proyecto (ver `reproducir.md`). Los márgenes ± son intervalos del 95 %. Todo lo que se ejecutó se informa, también lo que salió mal.

## 1. Datos y método en una página

- **Datos.** Cuatro acciones de EEUU (GOOGL, NVDA, XOM y PLD), velas diarias del 2 de enero de 2020 al 2 de noviembre de 2026 (1.718 velas cada una, con el mismo calendario), descargadas de Alpaca con precios ajustados. SPY y cuatro ETF sectoriales (XLK, XLC, XLE y XLRE) como referencia. BTCUSDT a 1 minuto (131.040 velas, enero a marzo de 2024) como banco de pruebas. Los 8-K de las cuatro empresas desde EDGAR, con la hora exacta de presentación: 353 eventos analizados y 295 independientes. Un dato erróneo (NVDA, 10 de junio de 2024) se detectó y corrigió.
- **Validación sin mirar el futuro (walk-forward).** Cada modelo se entrena solo con datos anteriores al día que predice y se reajusta periódicamente. Quedan unos 1.200 días de test por acción. Hay tests automáticos que comprueban que no se filtra información del futuro.
- **Margen de error.** El ± es un intervalo del 95 % de Newey-West, que tiene en cuenta que los errores de días seguidos se parecen entre sí.
- **Efecto de un evento.** El CAR (retorno anormal acumulado) es el retorno de la acción menos el que predice su relación con SPY en las 120 sesiones anteriores, sumado el día del evento y el siguiente. Para saber si un CAR es raro se compara con el de cientos de fechas al azar de la misma acción (placebo) mediante un test de permutaciones. Los avisos del mismo día se fusionan y los que están a menos de 5 sesiones de otro de mayor prioridad se descartan, para no contar dos veces la misma reacción.
- **Contrastes múltiples.** Con 5 categorías × 2 contrastes, un p resiste la corrección de Bonferroni solo si es menor de 0,005. Con las cuatro variantes del tono (4 contrastes cada una) el umbral es 0,0031.

## 2. La dirección del precio diario no se predice

Con unos 1.216 días de test por acción:

- Repetir el retorno de ayer (`persistence`) tiene un MAE entre un 40,7 % y un 44,0 % peor que predecir cero en las cuatro acciones, que es lo esperable si los retornos de días distintos son casi independientes (el error crece alrededor de un 41 %, la raíz de 2).
- La media histórica es prácticamente igual que cero (±0,2 %).
- El acierto de dirección coincide con el porcentaje de días de subida (GOOGL 52,2 %, NVDA 53,1 %, XOM 54,2 % y PLD 50,3 %), con un margen de ±2,8 puntos: ninguna se distingue de una moneda al aire.

**Lectura.** Con solo precios pasados, la dirección del día siguiente no es un objetivo realista. Esa es la vara que cualquier modelo tendría que superar.

## 3. La volatilidad sí se predice, de forma moderada

El rango diario (ln(máximo / mínimo)) tiene memoria: los días agitados siguen a otros agitados. Mejora del error medio (MAE) frente a la media histórica, con unos 1.217 días de test por acción:

| Modelo | GOOGL | NVDA | XOM | PLD |
|---|---|---|---|---|
| último valor | −21,5 ± 5,3 % | −1,5 ± 6,3 % | +9,3 ± 6,9 % | −4,7 ± 6,9 % |
| media de 5 días | +0,5 ± 4,2 % | +17,9 ± 5,7 % | +25,1 ± 5,3 % | +7,5 ± 5,9 % |
| media de 20 días | +2,0 ± 3,7 % | +14,6 ± 5,4 % | +25,2 ± 4,8 % | +8,4 ± 4,7 % |
| EWMA (0,94) | +4,0 ± 3,2 % | +15,8 ± 4,8 % | +27,4 ± 4,5 % | +11,4 ± 3,8 % |
| HAR | +4,8 ± 3,3 % | +18,4 ± 4,7 % | +26,0 ± 4,7 % | +11,7 ± 4,8 % |

- EWMA y HAR mejoran a la media histórica en las cuatro acciones, entre un 4 % y un 27 % menos de error, y empatan entre sí dentro del margen: la regresión (HAR) no compensa su complejidad frente a una media ponderada (EWMA).
- El rango de un solo día es demasiado ruidoso: «último valor» es peor que la media en tres de las cuatro acciones.
- Con QLIKE (una pérdida que castiga más subestimar el riesgo) el orden se mantiene: la mejora es de entre un 10 % y un 31 %, con márgenes amplios; es clara en NVDA (HAR +30,8 ± 12,5 %) y no concluyente en PLD (+18,3 ± 20,7 %).
- GOOGL mejora poco (4-5 %), quizá porque su volatilidad fue más estable en el periodo.

### 3.1 Mercado, sector y volumen no ayudan de forma consistente (HAR-X)

Se añadieron al HAR el rango de SPY, el del ETF sectorial y el volumen relativo. Mejora del MAE frente al HAR original:

| Variante | GOOGL | NVDA | XOM | PLD |
|---|---|---|---|---|
| mercado | +3,4 ± 1,3 % | +0,2 ± 1,2 % | +1,0 ± 2,6 % | +3,7 ± 1,9 % |
| sector | +4,2 ± 1,5 % | −1,0 ± 1,1 % | +1,9 ± 0,8 % | +4,9 ± 2,0 % |
| volumen | +0,6 ± 0,4 % | +0,1 ± 0,3 % | +0,4 ± 0,7 % | +0,6 ± 0,6 % |
| todo | +4,7 ± 1,7 % | −1,0 ± 1,4 % | +1,7 ± 2,7 % | +5,6 ± 2,4 % |

Mercado y sector mejoran algo el MAE en GOOGL y PLD, poco en XOM y nada en NVDA. Con QLIKE esas mejoras no se confirman (solo XOM mejora con el mercado: +8,3 ± 6,6 %), y el volumen no aporta. Ninguna variante gana de forma consistente en las cuatro acciones y con las dos métricas; con unas 80 comparaciones, algunas mejoras aisladas pueden ser casualidad.

### 3.2 Saber que mañana hay resultados sí ayuda (HAR-E)

HAR-E añade al HAR el calendario de resultados. La versión que se conserva (`har_enext`) usa una sola variable, «mañana hay resultados»; probar además «hoy hubo resultados» empeoró la predicción. Se usa la fecha real del 8-K como si se hubiera conocido de antemano, porque las empresas anuncian sus fechas de resultados con antelación. Error medio (MAE, en puntos porcentuales del rango diario) de las cuatro acciones juntas:

| Tipo de día | Días | HAR | HAR-E | Diferencia |
|---|---|---|---|---|
| víspera de resultados | 81 | 1,854 | 1,120 | HAR-E tiene un 39,6 % menos de error; HAR pierde un 65,6 ± 20,6 % frente a HAR-E |
| día de resultados | 81 | 0,883 | 0,882 | sin diferencia (−0,1 ± 0,5 %) |
| resto de días | 4.746 | 0,805 | 0,800 | menos del 1 % en ambos sentidos según la métrica |

**HAR-E es el mejor modelo de los probados.** La víspera tiene solo 81 días (unos 20 por acción): la mejora es grande, pero está medida con pocos casos.

### 3.3 Una red neuronal no supera a HAR-E

Un perceptrón multicapa (3 semillas, ventana de 22 días, entrenado con las cuatro acciones a la vez) que predice la corrección sobre la EWMA, con walk-forward, parada anticipada y GPU (61 segundos en total). Mejora frente a HAR-E (la referencia); un valor negativo significa que la red pierde:

| Muestra | Días | MAE de la red | QLIKE de la red |
|---|---|---|---|
| GOOGL | 1.227 | −4,1 ± 2,3 % | +3,5 ± 3,7 % |
| NVDA | 1.227 | −5,8 ± 2,8 % | −0,1 ± 3,7 % |
| XOM | 1.227 | −4,8 ± 2,8 % | +5,6 ± 7,0 % |
| PLD | 1.227 | −4,0 ± 3,2 % | +2,2 ± 5,1 % |
| conjunto, víspera de resultados | 81 | −13,0 ± 8,3 % | −30,3 ± 29,2 % |
| conjunto, día de resultados | 81 | −5,9 ± 9,7 % | −2,3 ± 18,0 % |
| conjunto, resto de días | 4.746 | −4,6 ± 1,5 % | +3,3 ± 2,5 % |

- La red es peor en MAE en las cuatro acciones (ningún margen incluye el 0) y queda empatada en QLIKE dentro del margen.
- En la víspera mejora mucho a HAR sin calendario (−13,0 % frente a −65,6 %) porque lee la variable de calendario, pero no alcanza a HAR-E.
- Explicación plausible, sin comprobar, de que MAE y QLIKE difieran: la red se entrena con QLIKE, que castiga más quedarse corto que pasarse, y tiende a predecir algo por encima.
- El único dato a favor (QLIKE +3,3 ± 2,5 % en el resto de días) es marginal: el margen del conjunto mezcla las cuatro acciones del mismo día, que están correlacionadas, y es algo optimista.

**Lectura.** Con unos 1.700 días por acción y cuatro acciones, una red no supera a un modelo lineal con las mismas variables. Es un resultado negativo válido: la complejidad adicional no se justifica con estos datos.

## 4. Los eventos mueven la acción en tamaño, no en dirección media

353 eventos analizados (21 omitidos por falta de datos antes o después), 295 independientes. Con 200.000 remuestreos, el p mínimo posible es 1/200.001, de ahí «< 0,00001».

| Categoría (8-K) | Eventos | CAR medio (± 95 %) | p | \|CAR\| medio | Azar | p | Rango real/previsto |
|---|---|---|---|---|---|---|---|
| directivos (5.02) | 61 | −0,22 ± 0,60 % | 0,44 | 1,67 % | 1,68 % | 0,50 | ×1,02 |
| otros | 26 | +0,28 ± 0,92 % | 0,54 | 1,69 % | 1,69 % | 0,49 | ×0,96 |
| otros eventos (8.01) | 64 | −0,48 ± 0,55 % | 0,058 | 1,57 % | 1,49 % | 0,32 | ×0,99 |
| divulgación a inversores (7.01, regfd) | 36 | +0,49 ± 1,20 % | 0,19 | 2,82 % | 1,77 % | < 0,00001 | ×1,21 |
| resultados (2.02) | 108 | +0,04 ± 0,95 % | 0,86 | 3,87 % | 1,64 % | < 0,00001 | ×1,65 |

- **Resultados trimestrales.** La reacción en tamaño es unas 2,4 veces la de un día normal (3,87 % frente a 1,64 %), con un rango un 65 % mayor que el previsto, pero sin dirección media (+0,04 ± 0,95 %): los buenos y los malos resultados se compensan. Las mayores reacciones coinciden con resultados conocidos (NVDA +19,6 % el 25 de mayo de 2023, GOOGL −12,9 % el 29 de octubre de 2026).
- **Categorías rutinarias** (directivos, otros y otros eventos): indistinguibles del azar. `otros_eventos` parecía negativo (p = 0,024) antes de fusionar avisos repetidos y filtrar solapes; con eventos independientes el p sube a 0,058. Era un artefacto.
- **Divulgación a inversores (regfd).** Pasa Bonferroni en conjunto, pero **no es robusta** (ver 4.1). Se cuenta como sugerente, no como hallazgo.
- **El mayor movimiento no resultado** (XOM −10,6 % el 3 de abril de 2025, un 8-K de directivos) coincide con la caída del petróleo y los aranceles de esa semana: probablemente no es efecto del 8-K, sino un golpe de sector que el modelo de mercado con SPY no recoge.

### 4.1 Robustez

Un p muy pequeño puede venir de una sola acción o de unos pocos eventos enormes. Para cada categoría que pasa el umbral (p < 0,005) se repite el contraste de magnitud solo con cada acción, sin cada acción y sin los 3 mayores eventos (esta última es conservadora: los quita de los eventos reales pero no de las fechas al azar).

**Resultados: robusta con excepciones.** Sin cada acción el cociente (|CAR| real entre el del azar) va de ×1,97 a ×2,87 con p < 0,00001, y sin los 3 mayores (NVDA 19,1 %, NVDA 12,3 % y GOOGL 11,7 %) queda ×2,18. Pero el efecto no es de todas las acciones:

| Acción | Eventos | \|CAR\| medio | Azar | Cociente | p |
|---|---|---|---|---|---|
| GOOGL | 27 | 5,00 % | 1,25 % | ×4,00 | < 0,00001 |
| NVDA | 27 | 5,63 % | 2,23 % | ×2,52 | < 0,00001 |
| PLD | 27 | 3,04 % | 1,28 % | ×2,37 | < 0,00001 |
| XOM | 27 | 1,81 % | 1,81 % | ×1,00 | 0,49 |

Los resultados de XOM no mueven la acción más que un día cualquiera. La causa no está explicada.

**regfd: frágil.** Pero 26 de sus 36 eventos son de XOM. Solo con XOM el cociente es ×1,60 (p = 0,0001); sin XOM quedan 10 eventos con ×1,59 y p = 0,029, que no pasa el umbral; sin los 3 mayores queda ×1,33 con p = 0,011, tampoco. El cociente es mayor que 1 en las cuatro acciones (GOOGL ×1,81, NVDA ×1,38, PLD ×1,87 y XOM ×1,60), pero GOOGL, NVDA y PLD tienen solo 3 o 4 eventos cada una y no se puede confirmar fuera de XOM.

## 5. El tono de los comunicados no predice la reacción

Para los comunicados de resultados se midió el tono con un lexicón financiero propio (cuenta palabras positivas y negativas) y con FinBERT (un modelo de lenguaje entrenado con textos financieros que puntúa cada frase), sobre el comunicado entero y solo sobre las citas entre comillas, que son las palabras del CEO. El tono relativo es el tono del comunicado menos la media de los anteriores de la misma empresa. Cuatro contrastes por variante y cuatro variantes: umbral de Bonferroni p < 0,0031.

| Variante | Comunicados | Spearman tono–CAR | \|tono\| frente a \|CAR\| | Terciles (pp) | Acierto de signo |
|---|---|---|---|---|---|
| lexicón, todo el texto | 97 | +0,162 (p = 0,119) | −0,194 (p = 0,060) | +2,11 (p = 0,105) | 60,0 % ± 11,1 |
| lexicón, solo citas | 75 | +0,177 (p = 0,132) | −0,282 (p = 0,0124) | +0,92 (p = 0,558) | 57,1 % ± 16,4 |
| FinBERT, todo el texto | 97 | +0,097 (p = 0,347) | −0,267 (p = 0,0096) | +0,43 (p = 0,762) | 60,8 % ± 9,7 |
| FinBERT, solo citas | 75 | −0,006 (p = 0,962) | −0,227 (p = 0,049) | −0,74 (p = 0,654) | 49,3 % ± 11,3 |

- **Ninguna variante supera el umbral** y no hay relación detectable entre el tono y la dirección de la reacción. FinBERT no mejora al lexicón, y sobre las citas no muestra nada.
- **Potencia limitada.** Con unos 100 eventos solo se distinguen del azar correlaciones de ±0,20 o más. «No se detecta efecto» no es lo mismo que «no hay efecto».
- **Un patrón sin explicar.** \|tono relativo\| frente a \|CAR\| sale negativo en las cuatro variantes (de −0,19 a −0,28; p entre 0,06 y 0,0096), al revés de lo que se esperaría. Ninguna pasa el umbral, no era una hipótesis previa, y las cuatro usan los mismos eventos y reacciones, así que no son cuatro confirmaciones.
- **Acierto de signo.** El 60,8 % de FinBERT (51,1 % a 70,5 %) apenas deja fuera el 50 % y es 1 de 16 contrastes: no se lee como señal.
- **FinBERT no es infalible.** En una comprobación con cuatro frases, puntuó como negativa «Operating costs fell sharply while margins reached a record high», que es buena noticia.

## 6. Síntesis

| Pregunta | Resultado | Evidencia | Confianza |
|---|---|---|---|
| ¿Se predice la dirección del retorno diario? | No | Acierto igual al de «siempre sube» (±2,8 pp) | Alta |
| ¿Se predice la volatilidad? | Sí, de forma moderada | EWMA y HAR: entre un 4 % y un 27 % menos de error que la media histórica | Media-alta (desigual por acción) |
| ¿Ayudan mercado, sector o volumen? | No de forma consistente | HAR-X, unas 80 comparaciones | Media |
| ¿Ayuda saber que mañana hay resultados? | Sí | HAR-E: un 39,6 % menos de error en la víspera (81 días) | Media (pocos días) |
| ¿Mejora una red neuronal? | No | MAE peor en las cuatro acciones, QLIKE empatado | Media-alta |
| ¿Mueven los resultados trimestrales la acción? | Sí en tamaño (×2,4), no en dirección media | 108 eventos, p < 0,00001; aguanta la robustez; no aparece en XOM | Alta |
| ¿Mueven otros 8-K la acción? | `regfd` sugerente pero frágil; el resto, no | Depende de XOM (26 de 36 eventos) | Baja |
| ¿Predice el tono la reacción? | No detectable | Cuatro variantes, ninguna pasa 0,0031 | Media (potencia limitada) |

## 7. Limitaciones

- **Sesgo de selección.** Son cuatro grandes empresas que existen hoy, elegidas con la ventaja de ver el pasado. No representan al mercado ni a las empresas que quebraron.
- **Periodo de test corto.** Unos 1.200 días por acción, y el mercado cambia de régimen: un resultado puede no repetirse.
- **Comparaciones múltiples.** Unas 80 comparaciones de modelos, 10 contrastes de eventos y 16 de tono. Los márgenes del 95 % de los modelos no están corregidos, así que varias mejoras aisladas pueden ser casualidad; se da más peso a lo que se repite en las cuatro acciones.
- **Pocos eventos.** 108 resultados (27 por acción), 36 de `regfd` y unos 100 comunicados con tono limitan lo que se puede detectar. Por eso se informa de la potencia y no solo del p.
- **Causalidad.** El CAR mezcla el evento con cualquier otra noticia de esos dos días (el caso de XOM en abril de 2025 lo ilustra). No hay datos intradía para aislar la reacción minuto a minuto.
- **Calendario de resultados.** HAR-E usa la fecha real del 8-K como si se hubiera conocido de antemano; es razonable porque las empresas anuncian sus fechas, pero es un supuesto.
- **Medida del tono.** El lexicón cuenta palabras sueltas, FinBERT se equivoca con algunas frases, y el extractor de citas solo encuentra las de 3 de 24 comunicados de PLD (el motivo no está aclarado). La variante de citas usa 75 de los 97 comunicados. No se dispone de las expectativas de los analistas, que son lo que mide de verdad la «sorpresa» de un resultado.
- **Datos gratuitos.** Alpaca ofrece datos consolidados con 15 minutos de retraso y un histórico intradía restringido.
- **Costes de operar** (comisiones, deslizamiento) sin modelar. Nada de esto es asesoramiento financiero.

## 8. Trabajo futuro

- Ampliar acciones, sectores y años para ganar potencia, y explicar por qué los resultados de XOM no mueven la acción.
- Incorporar las expectativas de los analistas (la sorpresa del resultado) y transcripciones de las llamadas con inversores.
- Datos intradía para medir la reacción minuto a minuto y reducir la contaminación por otras noticias.
- Probar modelos de lenguaje ajustados a este corpus y arreglar la extracción de citas de PLD.
- Figuras del mercado (operaciones de directivos en Form 4, posiciones 13F, ballenas on-chain) y una interfaz definitiva.

## Términos

- **8-K:** aviso que una empresa cotizada presenta a la SEC cuando ocurre algo relevante.
- **Bonferroni:** corrección que exige un p más pequeño cuando se hacen muchos contrastes a la vez.
- **CAR:** retorno anormal acumulado, es decir, lo que la acción se movió de más o de menos respecto a lo que predice el mercado, sumado en los días del evento.
- **EWMA:** media que da más peso a los días recientes. **HAR:** regresión que predice la volatilidad de mañana con el rango de hoy, el de la semana y el del mes. **HAR-E:** HAR con el calendario de resultados.
- **FinBERT:** modelo de lenguaje entrenado con textos financieros, que da la probabilidad de que una frase sea positiva, negativa o neutra.
- **MAE:** error medio absoluto. **QLIKE:** pérdida estándar para predicciones de volatilidad; castiga más subestimar el riesgo. Menor es mejor en ambas.
- **Newey-West:** margen de error que tiene en cuenta que los errores de días seguidos se parecen entre sí.
- **Placebo:** fechas elegidas al azar, sin evento, para ver cuánto se mueve la acción cuando no pasa nada.
- **Potencia:** capacidad de una prueba para detectar un efecto que existe; con pocos eventos es baja.
- **Test de permutaciones:** compara el efecto real con el de miles de fechas al azar para ver si podría ser casualidad.
- **Walk-forward:** validar entrenando con el pasado y probando en el futuro, ventana a ventana.
