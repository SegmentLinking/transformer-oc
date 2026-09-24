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

