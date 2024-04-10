'''
 The following three channel combinations are taken from the local-global-graph definitions in this article:
      - Author: Yi et al.
      - Year: 2023
      - Download URL: https://arxiv.org/abs/2105.02786
      - Reference:  Ding Y, Robinson N, Tong C, et al. LGGNet: Learning from local-global-graph representations for brain–computer interface[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.
      - Three types of local-global-graph definitions: GENERAL_REGION_LIST, FRONTAL_REGION_LIST, HEMISPHERE_REGION_LIST

'''
# can be used in DEAP, SEED, DREAMER dataset
GENERAL_REGION_LIST = [['FP1', 'FPZ', 'FP2'],
                        ['AF7', 'AF3', 'AFZ', 'AF4', 'AF8'],
                        ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
                        ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
                        ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'],
                        ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
                        ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
                        ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'], ['CB1','O1', 'OZ', 'O2','CB2'],
                        ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8']]  # 11 regions

FRONTAL_REGION_LIST = [['FP1', 'AF7', 'AF3'], ['FP2', 'AF4', 'AF8'],
                        ['F7', 'F5', 'F3', 'F1'], ['F2', 'F4', 'F6', 'F8'],
                        ['FC5', 'FC3', 'FC1'], ['FC2', 'FC4', 'FC6'],
                        ['FPZ', 'AFZ', 'FZ', 'FCZ'],
                        ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'],
                        ['CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6'],
                        ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
                        ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8'], ['CB1', 'O1', 'OZ', 'O2', 'CB2'],
                        ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8']]


HEMISPHERE_REGION_LIST = [['FP1', 'AF7', 'AF3'], ['FP2', 'AF4', 'AF8'],
                           ['F7', 'F5', 'F3', 'F1'], ['F2', 'F4', 'F6', 'F8'],
                           ['FC5', 'FC3', 'FC1'], ['FC2', 'FC4', 'FC6'],
                           ['C5', 'C3', 'C1'], ['C2', 'C4', 'C6'],
                           ['CP5', 'CP3', 'CP1'], ['CP2', 'CP4', 'CP6'],
                           ['P7', 'P5', 'P3', 'P1'], ['P2', 'P4', 'P6', 'P8'],
                           ['PO7', 'PO5', 'PO3', 'CB1', 'O1'], ['PO4', 'PO6', 'PO8', 'CB2', 'O2'],
                           ['FPZ', 'AFZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ'],
                           ['FT7', 'T7', 'TP7'],['FT8', 'T8', 'TP8']]

'''
   The following 1 brain region division comes from this article:
      - Author: Li et al.
      - Year: 2022
      - Download URL: https://ieeexplore.ieee.org/abstract/document/9765326
      - Reference:  Li, Yang, et al. GMSS: Graph-based multi-task self-supervised learning for eeg emotion recognition[J]. IEEE Transactions on Affective Computing, 2022.
      - One type of brain region division: NEIGHBOR_REGION_LIST
   
'''
NEIGHBOR_REGION_LIST = [['FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8'],
                           ['F7', 'F5', 'F3', 'FT7', 'FC5', 'FC3'],
                           ['F1', 'FZ', 'F2', 'FC1', 'FCZ', 'FC2'], ['F4', 'F6', 'F8', 'FC4', 'FC6', 'FT8'],
                           ['T7', 'C5', 'C3', 'TP7', 'CP5', 'CP3'],
                           ['C1', 'CZ', 'C2', 'CP1', 'CPZ', 'CP2', 'P1', 'PZ', 'P2'],
                           ['C4', 'C6', 'T8', 'CP4', 'CP6', 'TP8'], ['P7', 'P5', 'P3', 'PO7', 'PO5', 'CB1'],
                           ['PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2'], ['P4', 'P6', 'P8', 'PO6', 'PO8', 'CB2']]

POSTERIOR_REGION_LIST = [['FP1', 'FPZ', 'FP2'],
                    ['AF7', 'AF3', 'AFZ', 'AF4', 'AF8'],
                    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
                    ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6'],
                    ['CZ', 'CPZ', 'PZ', 'POZ', 'OZ'],
                    ['C5', 'C3', 'C1'], ['C2', 'C4', 'C6'], ['CP5', 'CP3', 'CP1'],
                    ['CP2', 'CP4', 'CP6'], ['P7', 'P5', 'P3', 'P1'], ['P2', 'P4', 'P6', 'P8'],
                    ['PO7', 'PO5', 'PO3', 'CB1', 'O1'], ['PO4', 'PO6', 'PO8', 'CB2', 'O2'], ['FT7', 'T7', 'TP7'],
                    ['FT8', 'T8', 'TP8']]  # 15 regions










