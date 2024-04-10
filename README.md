# Hierarchical Dynamic Local-Global-Graph Representation Learning for EEG Emotion Recognition

## Dataset
- SEED: 3 class (neutral, positive, negative)
- SEED-IV: 4 class (happiness, sadness, fear, neutral)
- SEED-V: 5 class (disgust, fear, sad, neutral, happy)
- DREAMER：2 class (valence, arousal)

<hr style="border-top: 3px solid black;">

### Load Dataset (Offline Transform, Feature Extraction ...)
Package:
- SEEDFeatureDataset()
- SEEDIVFeatureDataset()
- SEEDVFeatureDataset()
- DREAMERDataset()
  
 All Dataset loading method are based and extended on TorchEEG : https://github.com/torcheeg/torcheeg

- Segmentation (Raw Signal)
- Segmentation + DE (Differential Entropy)
- Segmentation + PSD (Power Spectral Density)

|          | dataset object         | Seg (channels, window)| Seg + DE (channels, bands) | Seg + PSD (channels,  bands) | Feature origin |
|----------|------------------------|-----------------------|----------------------------|----------------------------- |--------------- |
| SEED     | SEEDFeatureDataset()   |          -            | (62, 5)                     |        -                    | public      |
| SEED-IV  | SEEDIVFeatureDataset() |        -              | (62, 5)                     |      -                      | public        |
| SEED-V   | SEEDVFeatureDataset()  |           -            | (62, 5)                     |      -                       | public        |
| DREAMER  | DREAMERDataset()       | (14, 128)             | -                           | (14, 3)                      | private       |

EEG channels(num_electrodes), Segment size(Window size)

Feature origin：public（the public feature provided by dataset） and private (features extracted by this code )

### Split Dataset
train and test data are split by nest cross validation method for reliable generalization evaluation.
|           |SEED     | SEED-IV |  SEED-V  | DREAMER |
|-----------|---------|---------|----------|---------|
| inner loop|   2     |  2      |     2    |  3      |
| outer loop|   5     |  3      |     3    |  3      |

<hr style="border-top: 3px solid black;">

## Our Method
![Our Method](/src/model_graph.png)

### 1) Five local-global-graph definitions
![Our Method](/src/local_global_map.png)

For the SEED, SEED-IV, and SEED-V datasets, which are all 62-channel datasets, the same mask matrix and region list are used in the code.The DREAMER dataset uses another set of  mask matrix and region list.
Let's take the General graph definition as an example

|          | mask matrix         | region list| 
|----------|------------------------|-----------------------|
| SEED     | SEED_GENERAL_REGION_MASK_MATRIX    | SEED_GENERAL_REGION_LIST | 
| SEED-IV  | SEED_GENERAL_REGION_MASK_MATRIX    | SEED_GENERAL_REGION_LIST |
| SEED-V   | SEED_GENERAL_REGION_MASK_MATRIX    |  SEED_GENERAL_REGION_LIST | 
| DREAMER  | DREAMER_GENERAL_REGION_MASK_MATRIX | DREAMER_GENERAL_REGION_LIST  |

For each graph definition, we need to predefine for each dataset. For example, we run the code on the SEED dataset using the General graph definition method.
```python
python -m HDGNet --dataset_name=SEED --num_classes=3 --n_outer=5 --n_inner=2 --graph_defi=GENERAL
```
 we run the code on the DREAMER dataset using the General graph definition method. 
```python
python -m HDGNet --dataset_name=DREAMER --num_classes=2 --threshold=4.0 --emotion_key=valence --n_outer=3 --n_inner=3 --graph_defi=GENERAL --num_electrodes=14 --in_channels=3 --hid_channels=3 --out_channels=3
```
the  "GENERAL" in the "graph_defi=GENERAL" can be replaced with "FRONTAL", "HEMISPHERE", "NEIGHBOR", "POSTERIOR", then you can use the other four graph definitions.

### 2) Required dependencies
create environment
```
conda create -n eegpython3.7 python=3.7
conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
```


you can copy follow sentence into a txt file "environment.txt",

```
h5py==3.7.0
joblib==1.1.1
lmdb==1.3.0
pandas==1.3.5
scikit_learn==1.0.2
scipy==1.7.3
tqdm==4.64.1
```
Type ```pip install -r environment.txt``` at the command line and it will automatically pip install all packages listed in the environment.txt file.

Moreover, you also need to install the pytorch Geometric (Pyg), I recommend you to use ```conda install pyg -c pyg```.

Another thing you need to be aware of is you need to tell the Python interpreter that this directory is the root of your source code by setting the environment variable by follow code：
```
windows: set PYTHONPATH=your_directory/Affectiv_AI/;%PYTHONPATH%
linux: export PYTHONPATH=your_directory/Affectiv_AI/:$PYTHONPATH
```

### 3) Dataset storage
Please place the public dataset in the folder of "..\HDGNet\Affectiv_AI\examples\tmp_in".

### 4) Code demo
Please make sure you have followed the steps in the 2) and 3) to finsih the installation of required dependencies and datasets storage. 

Then, you need to go to the directory where the HDGNet.py file is located. 

Some examples.

SEED
```
python -m HDGNet --dataset_name=SEED --num_classes=3 --n_outer=5 --n_inner=2 --graph_defi=POSTERIOR
```

SEED-IV
```
python -m HDGNet --dataset_name=SEED-IV --num_classes=4 --n_outer=3 --n_inner=2 --graph_defi=POSTERIOR
```

SEED-V
```
python -m HDGNet --dataset_name=SEED-V --num_classes=5 --n_outer=3 --n_inner=2 --graph_defi=POSTERIOR
```

DREAMER
 - valance
```
python -m HDGNet --dataset_name=DREAMER --num_classes=2 --threshold=4.0 --emotion_key=valence --n_outer=3 --n_inner=3 --graph_defi=POSTERIOR --num_electrodes=14 --in_channels=3 --hid_channels=3 --out_channels=3
```
 - arousal
```
python -m HDGNet --dataset_name=DREAMER --num_classes=2 --threshold=4.0 --emotion_key=arousal --n_outer=3 --n_inner=3 --graph_defi=POSTERIOR --num_electrodes=14 --in_channels=3 --hid_channels=3 --out_channels=3
```


## Results
![Results](/src/result.png)

The output result is in the folder of "..\HDGNet\Affectiv_AI\examples\tmp_out".
