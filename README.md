# Deep Learning for Computer Vision

## Beschreibung
Dieses Repository enthält den Code für die Wartelinie-Erkennung auf SwissData Geo-Bildern mithilfe von Deep Learning für Computer Vision.

## GPU Server

Die aktuelle GPU Auslastung kann mittels `sudo nvtop` angezeigt werden. Die slurm Queue wird mittels `squeue` angezeigt und ein Job kann durch `scancel <JOBID>` gestoppt werden.

Um einen Slurm Job zu starten wird der folgende Befehl verwendet.

```shell
salloc -p students --time=2:00:00 -G a100:1 --ntasks=32 --mem-per-cpu=7G
```

Ist der Job an der Reihe (siehe squeue) kann mittels dem nachfolgenden Befehl eine Interaktive Apptainer Shell verwendet werden.

```shell
apptainer shell --nv "${HOME}/build-apptainer/tensorflow-2.16.1-gpu.sif"
```

Nun werden im Apptainer die entsprechenden Packete installiert.

```shell
pip install -r "${HOME}/VisionTransformer/requirements.txt"
```

Schlussendlich wird der Code ausgeführt.

```shell
python3 "${HOME}/VisionTransformer/CNN.py"
```