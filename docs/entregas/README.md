# Entregas del TFG

| Archivo | Qué es |
|---|---|
| `resultados.md` | Borrador del capítulo de resultados de la memoria: qué se midió, qué salió, con qué confianza y qué límites tiene. Cifras a 24 de noviembre de 2026. |
| `reproducir.md` | Anexo: cómo volver a obtener cada resultado, comando a comando, y qué hay que saber para que los números coincidan. |

- Son Markdown para copiar a la plantilla de la memoria (Word, LaTeX...). Las tablas y la lista de términos se pueden pegar tal cual.
- Las cifras salen de ejecutar el código del repositorio con los datos reales; no hay ninguna escrita a mano que no se pueda reproducir con `reproducir.md`.
- La documentación viva del proyecto (decisiones, módulos, glosario, cada paso) está en el Claude Doc «TFG Mercado: documentación del proyecto». Este directorio solo recoge lo que se entrega.
- `tests/test_entregas.py` comprueba que los comandos de `reproducir.md` apuntan a módulos y opciones que existen, para que el anexo no se quede obsoleto si cambia el código.
