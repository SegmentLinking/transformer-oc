#!/usr/bin/env bash
#
# Split each tracking ntuple into several smaller LSTNtuples.
#
# Each tracking ntuple is processed in NCHUNKS passes using lst's own
# job splitting (-j NCHUNKS -I CHUNK), so no pass holds more than
# EVENTS_PER_LSTNTUPLE events in memory or in its output file.
#
# Usage: ./make_lstntuples.sh [i_trackingNtuple] [j_LSTNtuple]
#   no args:  process every chunk of every tracking ntuple
#   i:        process every chunk of trackingNtuple_<i>.root
#   i j:      process only chunk j (0-based) of trackingNtuple_<i>.root
# Safe to rerun: chunks whose output already exists are skipped.

LST=lst_cpu
CMSSW_SRC=/ceph/users/atuna/CMSSW_20_1_0_pre2/src
START_DIR=${PWD}

if ! command -v "${LST}" > /dev/null; then
    echo "[env ] ${LST} not found, setting up from ${CMSSW_SRC}"
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    cd "${CMSSW_SRC}"
    eval "$(scramv1 runtime -sh)" # what the `cmsenv` alias does
    source RecoTracker/LSTCore/standalone/setup.sh
    cd "${START_DIR}"
fi

set -euo pipefail

if ! command -v "${LST}" > /dev/null; then
    echo "${LST} still not found after environment setup" >&2
    exit 1
fi

I_TRACKING_NTUPLE=${1:-*}
J_LSTNTUPLE=${2:-}

INPUT_DIR=/ceph/users/atuna/CMSSW_15_1_0_pre4/src/ttbar/n1e3_PU200
OUTPUT_DIR="."
# OUTPUT_DIR=output
LOG_DIR=${OUTPUT_DIR}/logs
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

EVENTS_PER_TRACKING_NTUPLE=1000
EVENTS_PER_LSTNTUPLE=2 # 100
NCHUNKS=$(( EVENTS_PER_TRACKING_NTUPLE / EVENTS_PER_LSTNTUPLE ))
if (( EVENTS_PER_TRACKING_NTUPLE % EVENTS_PER_LSTNTUPLE != 0 )); then
    echo "EVENTS_PER_TRACKING_NTUPLE must be a multiple of EVENTS_PER_LSTNTUPLE" >&2
    exit 1
fi

shopt -s nullglob
CHUNK_FIRST=${J_LSTNTUPLE:-0}
CHUNK_LAST=${J_LSTNTUPLE:-$(( NCHUNKS - 1 ))}

LST_FLAGS=(--md --ls --t3 --t3dnn --t4 --t5 -v 1)

# I_TRACKING_NTUPLE is deliberately unquoted: it is either an index or "*"
TRACKING_NTUPLES=("${INPUT_DIR}"/trackingNtuple_${I_TRACKING_NTUPLE}.root)
if (( ${#TRACKING_NTUPLES[@]} == 0 )); then
    echo "No tracking ntuples found in ${INPUT_DIR}" >&2
    exit 1
fi

for TRACKING_NTUPLE in "${TRACKING_NTUPLES[@]}"; do

    # trackingNtuple_3.root -> 3
    TAG=$(basename "${TRACKING_NTUPLE}" .root)
    TAG=${TAG#trackingNtuple_}

    for (( CHUNK = CHUNK_FIRST; CHUNK <= CHUNK_LAST; CHUNK++ )); do

        NAME=LSTNtuple_${TAG}_${CHUNK}
        OUTPUT=${OUTPUT_DIR}/${NAME}.root
        PARTIAL=${OUTPUT_DIR}/${NAME}.partial.root
        LOG=${LOG_DIR}/${NAME}.log

        if [[ -f "${OUTPUT}" ]]; then
            echo "[skip] ${OUTPUT} already exists"
            continue
        fi

        echo "[run ] ${TRACKING_NTUPLE} chunk $(( CHUNK + 1 ))/${NCHUNKS} -> ${OUTPUT}"

        # Write to a .partial file and rename on success, so an interrupted
        # job never leaves behind something that looks like a finished chunk.
        "${LST}" \
            -i "${TRACKING_NTUPLE}" \
            -o "${PARTIAL}" \
            -j "${NCHUNKS}" \
            -I "${CHUNK}" \
            "${LST_FLAGS[@]}" \
            &> "${LOG}"

        mv "${PARTIAL}" "${OUTPUT}"

    done
done

echo "Done. Outputs in ${OUTPUT_DIR}/"
