# Transformer OC

# 1) Data

Graph datasets (one graph per event) are made with [transformer-datasets](https://github.com/SegmentLinking/transformer-datasets), which documents the pipeline from simulation to graphs and the graph format. For example:

```
python -m transformer_datasets.t3_processing --input LSTNtuple_*.root --output data/pu200_t3/ --seed 42 -n 16
```

This writes `train/` and `val/` graph directories. Put them under `data/` and point `train_data_dir` and `val_data_dir` in the config at them.

# 2) Training

Train either locally or on the NRP cluster. Locally you can run the project's training entrypoint (see `src/main.py`). To run on NRP, use the deployment YAMLs under `misc/nrp/`.

```
python3 src/main.py --config config_pu200.yaml
```

See config_pu200.yaml for example.

# 3) Analysis

After training, run the analysis tools to evaluate or plot results:

```
python3 analysis/main.py --config config_pu200.yaml --output output/lightning_logs/version_6/ --n-events 1000 --epsilon 0.05
```

The plots are stored in the same lightning_logs dir as the trained model checkpoints.
