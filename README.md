# dubalign

Detecta y corrige desfases entre la pista de audio original (japonés) y los
doblajes/subtítulos de un mismo archivo, o añade un audio de otro release.
Los originales nunca se modifican: los corregidos van a `output/`.

## Interfaz web (Docker)

```bash
cp .env.example .env          # opcional: puerto y carpetas
docker compose up -d --build  # http://localhost:7341
docker compose down
```

1. Deja los episodios en `media/` (se monta en solo lectura).
2. **Analizar todo**: mide cada audio y subtítulo contra la pista de referencia.
3. Elige un episodio:
   - tabla de audios y subtítulos con el desfase medido y la corrección propuesta (editable);
   - **Episodio completo** (original o archivo guardado):
     - mapa de sincronía: el desfase medido cada 2 s en todo el episodio; los huecos son
       tramos sin nada común que medir (solo diálogo o silencio);
     - espectrogramas de todas las pistas, volumen superpuesto y subtítulos;
       arrastrar = moverse, rueda = zoom (del episodio entero a fracciones de segundo),
       clic = cursor;
     - reproducción continua: japonés en el oído izquierdo y la pista elegida en el
       derecho (o la pista sola). Espacio = reproducir/pausa, ←/→ = 2 s;
     - la dirección de la página guarda la posición (`#archivo@segundo+ancho`, `:o` = original)
       para volver al mismo punto;
   - detalle en imagen con **Original / Con la corrección (simulado) / Archivo guardado**,
     para probar una corrección antes de guardarla.
4. **Guardar**: escribe `output/<archivo>`, lo vuelve a medir y lo deja listo en el visor.

Informes, imágenes y verificaciones se guardan en `work/`.

## Línea de comandos (venv local)

```bash
./run.sh -h                                   # crea .venv la primera vez
.venv/bin/python fix.py media/*.mkv --dry-run # informe
.venv/bin/python fix.py media/*.mkv           # corrige lo necesario en output/
.venv/bin/python spectro.py media/X.mkv --start 760 --dur 4
./run.sh ref.mkv otro_release.mkv --lang spa  # añadir un audio de otro archivo
```

## Qué corrige

| Caso | Cómo |
|---|---|
| Desfase constante | `mkvmerge --sync`, sin recodificar |
| Cortes (tramos con delay distinto, p. ej. falta medio segundo a mitad o el avance final difiere) | recoloca cada tramo y recodifica **solo esa pista** (AAC o FLAC) |
| Velocidad distinta (25 / 24 / 23.976 fps) | `atempo` y recodifica esa pista |
| Subtítulos | `--sync` sobre la pista de subtítulos |

### Límites conocidos

- Desfases de audio < 40 ms se consideran correctos (un fotograma son ~42 ms).
- Si el audio es de otro episodio, sale "no medible" y no se toca nada. Para no
  confundirse con el opening/ending (iguales en todos los episodios), las
  coincidencias tienen que estar repartidas por todo el episodio, al menos una
  por cada tercio.
- Subtítulos: se miden contra los momentos con voz (diferencia entre doblajes). Que
  vayan 100–250 ms por delante de la voz es normal y no se corrige. Si un subtítulo
  coincide con un doblaje desfasado (p. ej. "English CC" hecho para el doblaje
  inglés), hereda su mismo delay.
- Saltos de menos de 150 ms en el primer/último minuto (avances, intros) se ignoran
  para no recodificar una pista por algo imperceptible.
