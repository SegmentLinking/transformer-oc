#!/usr/bin/env bash
# Builds run_torchscript against CMSSW_20_1_0_pre3's libtorch (torch 2.13, CUDA build).
# usage: source pyenv-cmssw/activate.sh (or cmsenv) && bash export/cpp/build.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH=/cvmfs/cms.cern.ch/el8_amd64_gcc14/external/py3-torch-cuda/2.13.0-0920f341c2e05aaa43b0466c8d075454
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
# --disable-new-dtags: RPATH (not RUNPATH) wins over LD_LIBRARY_PATH, so cmsenv's CPU-only py3-torch libs are not picked up
g++ -std=c++20 -O2 "$HERE/run_torchscript.cc" -o "$HERE/run_torchscript" \
    -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
    -L"$TORCH/lib" -Wl,--disable-new-dtags -Wl,-rpath,"$TORCH/lib" \
    -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed \
    -L"$CUDA_HOME/lib64" -lcudart
echo "built $HERE/run_torchscript"
