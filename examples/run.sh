#!/bin/bash

# Select virtual environment 
HOST=`hostname`
echo Host $HOST
if [[ $HOST == f428d* ]]
then
	echo Running in workstation environment
	source /ereborfs/zenkfrie/venvs/default/bin/activate # TODO update personalized virtual env paths
else
	echo Running in GPU cluster environment
	source /tungstenfs/scratch/gzenke/shared/venvs/gpuc/bin/activate # TODO update personalized path
fi

TUNGSTEN="/tungstenfs/scratch/gzenke/"
RESDIR="$HOME/data/SpikingCNNs/rawhd/deep/"
mkdir -p $RESDIR


# DATASET="--training_data $TUNGSTEN/datasets/speech_commands \
#  --dataset sc --nb_classes 35"

DATASET="--training_data $TUNGSTEN/datasets/RawHeidelbergDigits \
 --dataset hd --nb_classes 20"

ARGS="$DATASET \
 --batch_size 100 --gpu --nb_workers 2 --verbose --dir $RESDIR \
 --beta 20 --lr 5e-3 \
 --loss SumOverTime \
 --validation_split 0.9 \
 --encoder_fanout 10 \
 --encoder_gain 1.0 \
 --lowerBoundL2Thr 1e-3 \
 --perNeuronUpperBound 60
 --upperBoundL2Thr 50 \
 --upperBoundL1Thr 20 \
 --lowerBoundL2Strength 100.0 --upperBoundL2Strength 0.1 --upperBoundL1Strength 0.1 \
 --channel_fanout 1.0 \
 --tau_readout 20e-3 \
 --tau_mem 20e-3 \
 --tau_syn 5e-3 \
 --nb_priming 0 --nb_epochs 30 --fine_tune 0 --verbose \
 --duration 0.4 --time_scale 1.0 --time_step 2e-3 \
 --plot --stp --optimizer SMORMS4 \
 --wandbentity fzenke --wandbproj example"


for REP in 0 1 2; do
		SHARED="$ARGS --nb_hidden_blocks 1 --seed 142$REP --wscale 3.0"
		CUDA_VISIBLE_DEVICES=1 python sim_raw.py $SHARED --prefix ctl 
	done
done



