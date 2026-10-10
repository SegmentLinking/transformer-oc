#!/usr/bin/env bash
#
# Split each tracking ntuple into several smaller LSTNtuples.
#
# Each tracking ntuple is processed in NCHUNKS passes using lst's own
# job splitting (-j NCHUNKS -I CHUNK), so no pass holds more than
# EVENTS_PER_LSTNTUPLE events in memory or in its output file.
#
# Usage: ./make_lstntuples.sh [i_trackingNtuple] [j_LSTNtuple]
#   i j: process chunk j (0-based) of trackingNtuple_<i>.root
# Safe to rerun: chunks whose output already exists are skipped.
#
# Run everything in a loop like:
# for I in {1..7}; for J in {0..9}; do ./make_lstntuples.sh $I $J; done; done

set -eo pipefail

LST=lst_cpu
START_DIR=${PWD}
CMSSW_SRC=/ceph/users/atuna/CMSSW_20_1_0_pre2/src

if ! command -v "${LST}" > /dev/null; then
    echo "[env ] ${LST} not found, setting up from ${CMSSW_SRC}"
    source /cvmfs/cms.cern.ch/cmsset_default.sh || true
    cd "${CMSSW_SRC}"
    eval "$(scramv1 runtime -sh)" || true # what the `cmsenv` alias does
    source RecoTracker/LSTCore/standalone/setup.sh || true
    cd "${START_DIR}"
fi
if ! command -v "${LST}" > /dev/null; then
    echo "${LST} still not found after environment setup" >&2
    exit 1
fi
if [ -z "$1" ]; then
    echo "Need to provide I_TRACKINGNTUPLE"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Need to provide J_LSTNTUPLE_CHUNK"
    exit 1
fi

set -euo pipefail

I_TRACKINGNTUPLE=${1}
J_LSTNTUPLE=${2}

LST_FLAGS=(--md --ls --t3 --t3dnn --t4 --t5 -v 1)
INPUT_DIR=/ceph/users/atuna/CMSSW_15_1_0_pre4/src/ttbar/n1e3_PU200
EVENTS_PER_TRACKINGNTUPLE=1000
EVENTS_PER_LSTNTUPLE=2 # 100
NCHUNKS=$(( EVENTS_PER_TRACKINGNTUPLE / EVENTS_PER_LSTNTUPLE ))

TRACKINGNTUPLE=${INPUT_DIR}/trackingNtuple_${I_TRACKINGNTUPLE}.root
NAME=LSTNtuple_${I_TRACKINGNTUPLE}_${J_LSTNTUPLE}
LSTNTUPLE=${NAME}.root
PARTIAL=${NAME}.partial.root
LOG=${NAME}.txt

if [[ -f "${LSTNTUPLE}" ]]; then
    echo "Skipping ${LSTNTUPLE}, already exists"
    continue
fi

echo "Converting ${TRACKINGNTUPLE} into ${LSTNTUPLE} with ${NCHUNKS} chunks..."

# Write to a .partial file and rename on success, so an interrupted
# job never leaves behind something that looks like a finished chunk.
${LST} \
    -i ${TRACKINGNTUPLE} \
    -o ${PARTIAL} \
    -j ${NCHUNKS} \
    -I ${J_LSTNTUPLE} \
    "${LST_FLAGS[@]}" \
    &> ${LOG}

mv ${PARTIAL} ${LSTNTUPLE}

