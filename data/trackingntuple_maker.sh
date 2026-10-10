# 29834.1 TTbar_14TeV_TuneCP5_Run4D110PU_GenSimHLBeamSpot14+DigiTriggerPU_Run4D110PU+RecoGlobalPU_trackingOnly_Run4D110PU+HARVESTGlobalPU_trackingOnly_Run4D110PU

NPROCS=15
NEVENTS=1000
NTHREADS=20

# time for i in $(seq 1 ${NPROCS}); do
#     time cmsDriver.py TTbar_14TeV_TuneCP5_cfi \
#          -s GEN,SIM \
#          -n ${NEVENTS} \
#          --conditions auto:phase2_realistic_T33 \
#          --beamspot DBrealisticHLLHC \
#          --datatier GEN-SIM \
#          --eventcontent FEVTDEBUG \
#          --geometry ExtendedRun4D110 \
#          --era Phase2C17I13M9 \
#          --relval 9000,100 \
#          --nThreads ${NTHREADS} \
#          --fileout file:TTbar_GEN_SIM_$i.root \
#          --python_filename ttbar_GENSIM_$i.py \
#          --customise_commands "process.RandomNumberGeneratorService.generator.initialSeed = $((1000 + i))" \
#          --no_exec
# done

# for i in $(seq 1 ${NPROCS}); do
#   time cmsRun ttbar_GENSIM_$i.py # &
# done
# wait

# for i in $(seq 1 ${NPROCS}); do
#     time cmsDriver.py step2 \
#          -s DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:@relvalRun4 \
#          --conditions auto:phase2_realistic_T33 \
#          --datatier GEN-SIM-DIGI-RAW \
#          -n ${NEVENTS} \
#          --eventcontent FEVTDEBUGHLT \
#          --geometry ExtendedRun4D110 \
#          --era Phase2C17I13M9 \
#          --pileup AVE_200_BX_25ns \
#          --pileup_input filelist:/ceph/cms/store/user/slava77/samples/RelValMinBias_14TeV/CMSSW_15_1_0_pre1-141X_mcRun4_realistic_v3_STD_MinBias_Run4D110_GenSim-v1/GEN-SIM/files.txt \
#          --filein file:TTbar_GEN_SIM_$i.root \
#          --fileout file:step2_$i.root \
#          --python_filename ttbar_step2_$i.py \
#          --nThreads ${NTHREADS} # --no_exec
# done

# ANCHOR='process.mix.digitizers = cms.PSet()'
# NEWTEXT='process.simHitTPAssocProducer.simHitSrc += ["g4SimHits:TrackerHitsPixelBarrelHighTof", "g4SimHits:TrackerHitsPixelEndcapHighTof"]'

# for i in $(seq 1 ${NPROCS}); do
#     time cmsDriver.py step3 \
#          -s RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM \
#          --conditions auto:phase2_realistic_T33 \
#          --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO \
#          -n ${NEVENTS} \
#          --eventcontent FEVTDEBUGHLT,MINIAODSIM,DQM \
#          --geometry ExtendedRun4D110 \
#          --era Phase2C17I13M9 \
#          --pileup AVE_200_BX_25ns \
#          --pileup_input filelist:/ceph/cms/store/user/slava77/samples/RelValMinBias_14TeV/CMSSW_15_1_0_pre1-141X_mcRun4_realistic_v3_STD_MinBias_Run4D110_GenSim-v1/GEN-SIM/files.txt \
#          --nThreads ${NTHREADS} \
#          --lazy_download \
#          --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtuple \
#          --customise Validation/RecoTrack/customiseTrackingNtuple.extendedContent \
#          --customise_command="process.trackingNtuple.clusterMasks = []" \
#          --filein  file:step2_$i.root \
#          --fileout file:step3_$i.root \
#          --python_filename ttbar_step3_$i.py \
#          --no_exec
#     sed -i "s/${ANCHOR}/${ANCHOR}\n${NEWTEXT}/g" ttbar_step3_$i.py
# done

for i in $(seq 12 ${NPROCS}); do
    time cmsRun ttbar_step3_$i.py 2>&1 | tee trackingNtuple.log
    mv trackingNtuple.root trackingNtuple_$i.root
done

