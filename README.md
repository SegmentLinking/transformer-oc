# Transformer OC

# 0) Environment

Run everything inside the `ml-tracking` image, which has torch and other libraries. Building it yourself is optional: the Dockerfile is in `misc/nrp/train/`. You can download it locally with:

```
export APPTAINER_TMPDIR=${HOME}/apptainer/tmp
export APPTAINER_CACHEDIR=${HOME}/apptainer/cache
mkdir -p ${APPTAINER_TMPDIR} ${APPTAINER_CACHEDIR}
apptainer build ml-tracking.sif docker://alexandertuna/ml-tracking:latest
```

# 1) Data processing

Run one of the preprocessing scripts to create graph datasets (one graph per event). For example:

```
python3 t3_processing.py --input ./CMSSW_15_1_0_pre2/src/RecoTracker/LSTCore/standalone/output_pu200_* --output pu200_t3/ -n 4
```

Each script will produce training and validation graph datasets (one file per event) in the script's configured output location.

# 2) Training

Train either locally or on the NRP cluster. Locally you can run the project's training entrypoint (see `src/main.py`). To run on NRP, use the deployment YAMLs under `misc/nrp/`.

```
python3 main.py --config config.yaml
```

See config_pu200.yaml for example.

# 3) Analysis

After training, run the analysis tools to evaluate or plot results:

```
python3 analysis/main.py --config config_pu200.yaml --output output/lightning_logs/version_6/ --n-events 1000 --epsilon 0.05
```

The plots are stored in the same lightning_logs dir as the trained model checkpoints.