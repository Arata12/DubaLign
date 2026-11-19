# dubalign

Detecta y corrige desfases entre la pista de audio original (japonés) y los
doblajes/subtítulos de un mismo archivo, o añade un audio de otro release.
Motor de análisis DSP, generación de espectrogramas y servidor web implementados
en **Rust** (`realfft`, `rayon`, `axum`, `tiny-skia`), ejecutados íntegramente en
Docker junto a `ffmpeg` y `mkvtoolnix`.
Los originales nunca se modifican: los corregidos van a `output/`.

## Interfaz web (Docker)

```bash
cp .env.example .env          # opcional: puerto y carpetas
docker compose up -d --build  # http://localhost:7341
docker compose down
```

1. Deja los episodios en `media/` (sueltos o organizados por carpetas/subcarpetas, p. ej. `media/bungo s1/`, `media/bungo s2/` — se monta en solo lectura).
2. **Analizar todo** / **Procesar todo**, o usa los botones de cada carpeta (**Analizar** / **Procesar** para analizar y guardar automáticamente en `output/<carpeta>/` los episodios que necesiten corrección; si todos ya estaban guardados, vuelve a procesarlos).
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

## Línea de comandos (Docker)

```bash
./run.sh -h                                           # ayuda de dubalign-sync
./run.sh ref.mkv otro_release.mkv --lang spa          # añadir un audio de otro archivo
docker compose run --rm --entrypoint dubalign-fix web /media/*.mkv --dry-run
docker compose run --rm --entrypoint dubalign-fix web "/media/bungo s1"
docker compose run --rm --entrypoint dubalign-spectro web /media/X.mkv --start 760 --dur 4 -o /output/X.png
```

## Qué corrige

| Caso | Cómo |
|---|---|
| Desfase constante | `mkvmerge --sync`, sin recodificar |
| Pistas/doblajes más largos que el audio original | Recorte sin recodificar con `ffmpeg -t` (audios y subtítulos largos) para evitar audio sobre pantalla negra; respaldo PCM 32 bits → FLAC si el códec no admite copia directa |
| Cortes (tramos con delay distinto, p. ej. falta medio segundo a mitad o el avance final difiere) | Primero **duplicación y dropeo de frames sin recodificar** (`AAC-LC`, `AC-3`, `E-AC-3` independiente); cuando no es posible, salto directo a **PCM 32 bits → FLAC compresión máxima (`-compression_level 12`)** |
| Velocidad distinta (25 / 24 / 23.976 fps o deriva de reloj) | Muestra **FPS referencia** vs **FPS de la pista**, detecta si el tono musical está alterado (p. ej. *speedup* PAL +0.72 semitonos) o ya compensado, y recodifica en **PCM 32 bits flotante** corrigiendo tono+velocidad (`asetrate` + `soxr` 28-bit) o solo velocidad (`atempo`) → **FLAC compresión máxima (`-compression_level 12`)** |
| Subtítulos | Diálogo normal con `--sync` (nunca pisado por el doblaje si ya cuadra con japonés); pistas **Signs / Forced** no medibles heredan automáticamente el `offset`, los `cortes` por tramos (`.ass`, `.srt`, `PGS`, `VobSub`) o el `tempo` del audio de su idioma |

### Límites conocidos

- Desfases de audio < 40 ms se consideran correctos (un fotograma son ~42 ms).
- Si el audio es de otro episodio, sale "no medible" y no se toca nada. Para no
  confundirse con el opening/ending (iguales en todos los episodios), las
  coincidencias tienen que estar repartidas por todo el episodio, al menos una
  por cada tercio.
- Subtítulos: los normales se miden contra los momentos con voz (diferencia entre doblajes). Que
  vayan 100–250 ms por delante de la voz es normal y no se corrige. Las pistas *Signs / Forced*
  sin voz medible heredan el ajuste (`offset`, `cortes` o `tempo`) del doblaje de su mismo idioma,
  y los subtítulos *CC* hechos sobre un doblaje desfasado ajustan su delay al de ese audio.
- Saltos de menos de 150 ms en el primer/último minuto (avances, intros) se ignoran
  para no recodificar una pista por algo imperceptible.

## Licencia

MIT, ver [LICENSE](LICENSE).
