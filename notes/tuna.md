# Notes

## Data

The primary dataset is ttbar PU200. This is first generated from scratch as a tracking ntuple, then converted into a LSTNtuple to be pre-processed by transformer-oc.

Alex generated ttbar tracking ntuples is here:

```
# Code to generating tracking ntuples
/ceph/users/atuna/CMSSW_15_1_0_pre4/src/ttbar/n1e3_PU200/workflow.sh

# One of the tracking ntuples
/ceph/users/atuna/CMSSW_15_1_0_pre4/src/ttbar/n1e3_PU200/trackingNtuple_1.root
```

Aashay converted these to LSTNtuples. They can be found in a NRP PVC here:

```
kubectl -n cern-cms-gpu-tracking get pvc
kubectl -n cern-cms-gpu-tracking apply -f misc/nrp/data/data.yaml
kubectl -n cern-cms-gpu-tracking exec -it deploy/ml-tracking-data -- ls /data/input/output_pu200
```

Or on uaf3 here:

```
/home/users/aaarora/phys/tracking/lst/lstod/CMSSW_15_1_0_pre2/src/RecoTracker/LSTCore/standalone/output_pu200/
```

And in case Aashay's home directory gets expunged, Alex made a copy here:

```
/ceph/users/atuna/work/cms_tracking_ml/data/output_pu200/
```

## Environment

Alex is running on uaf3 for now.

Start a new environment like:

```
# NB: I might need "torch==2.5.1" for compatibility with FastGraphCompute
# Skipping that for now for simplicity
cd ~/work/cms_tracking_ml/
python3.12 -m venv venv
source venv/bin/activate
pip install torch torch_geometric uproot awkward pandas numpy
du -hs venv # 5.5GB
```

Start the existing environment like:

```
source venv/bin/activate
```

## Pre-processing

This step converts LSTNtuples into torch tensors saved on disk. You can run on one event quickly like:

```
cd transformer-oc/data/
python t3_processing.py --seed 42 --input ../../data/output_pu200/output_pu200_0.root --n_events 1
```

Aashay's pre-processing is available in the same PVC:

```
kubectl -n cern-cms-gpu-tracking exec -it deploy/ml-tracking-data -- du -hs /data/pu200_t3/train /data/pu200_t3/val
53G	/data/pu200_t3/train
13G	/data/pu200_t3/val
```

