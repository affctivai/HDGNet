# Hierarchical Dynamic Local-Global-Graph Representation Learning for EEG Emotion Recognition

## Dataset
- SEED: 3 class (neutral, positive, negative)
- SEED-IV: 4 class (happiness, sadness, fear, neutral)
- SEED-V: 5 class (disgust, fear, sad, neutral, happy)
- DREAMER：2 class (valence, arousal)

# HDGNet
Hierarchical Dynamic Local-Global-Graph Representation Learning

# Train
the train file is the Affectiv_AI/examples/HDGNet.py

# reference
if you not need to split dataset, please add "--Split=False"

SEED

python -m HDGNet --dataset_name=SEED --num_classes=3 --n_outer=5 --n_inner=2 --graph_defi=POSTERIOR

SEED-IV

python -m HDGNet --dataset_name=SEED-IV --num_classes=4 --n_outer=3 --n_inner=2 --graph_defi=POSTERIOR

SEED-V

python -m HDGNet --dataset_name=SEED-V --num_classes=5 --n_outer=3 --n_inner=2 --graph_defi=POSTERIOR

DREAMER

valance

python -m HDGNet --dataset_name=DREAMER --num_classes=2 --threshold=4.0 --emotion_key=valence --n_outer=3 --n_inner=3 --graph_defi=POSTERIOR --num_electrodes=14 --in_channels=3 --hid_channels=3 --out_channels=3

arousal

python -m HDGNet --dataset_name=DREAMER --num_classes=2 --threshold=4.0 --emotion_key=arousal --n_outer=3 --n_inner=3 --graph_defi=POSTERIOR --num_electrodes=14 --in_channels=3 --hid_channels=3 --out_channels=3
