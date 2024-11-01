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
python3 "${HOME}/VisionTransformer/pythonScript/CNN.py"
```

### Neuer Sweep erstellen

```shell
python3 "${HOME}/VisionTransformer/pythonScript/modules/create_sweep.py"
```

Das Script erstellt einen neuen Sweep mit dem angegebenen Namen und gibt die benötigte Sweep URL aus. Die generierte Sweep ID (`silvan-wiedmer-fhgr/VisionTransformer/<sweep-id>`) muss im Anschliesenden Befehl eingetragen werden.

### Agent auf Sweep Anwenden

```shell
python3 "${HOME}/VisionTransformer/pythonScript/CNN.py" <sweep-id>
```

### Scripts

Um das starten von mehreren Agenten zu vereinfachen, enthälte der Ordner `Scripts` für jedes Modell ein Skript um die Agenten auf einen Sweep anzuwenden. Dazu muss der folgende Befeh ausgeführt werden.

```shell
bash Scripts/start_<model_name>_sweep_agents.sh <sweep-id>
```