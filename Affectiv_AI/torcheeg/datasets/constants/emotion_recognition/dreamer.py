import numpy as np

from ..region_1020 import (FRONTAL_REGION_LIST,
                           GENERAL_REGION_LIST,
                           HEMISPHERE_REGION_LIST,
                           NEIGHBOR_REGION_LIST,
                           POSTERIOR_REGION_LIST,
                           )

from ..utils import (format_adj_matrix_from_adj_list,
                     format_region_channel_list,
                     format_region_adjacent_list)

DREAMER_CHANNEL_LIST = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4',
    'F8', 'AF4'
]

DREAMER_LOCATION_LIST = [['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                         ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
                         ['F7', '-', 'F3', '-', '-', '-', 'F4', '-', 'F8'],
                         ['-', 'FC5', '-', '-', '-', '-', '-', 'FC6', '-'],
                         ['T7', '-', '-', '-', '-', '-', '-', '-', 'T8'],
                         ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                         ['P7', '-', '-', '-', '-', '-', '-', '-', 'P8'],
                         ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
                         ['-', '-', '-', 'O1', '-', 'O2', '-', '-', '-']]

DREAMER_ADJACENCY_LIST = {
    'AF3': ['F3', 'AF4'],
    'AF4': ['AF3', 'F4'],
    'F7': ['F3', 'FC5', 'T7'],
    'F3': ['AF3', 'F7', 'FC5'],
    'F4': ['AF4', 'F8', 'FC6'],
    'F8': ['F4', 'FC6', 'T8'],
    'FC5': ['F7', 'F3', 'T7'],
    'FC6': ['F4', 'F8', 'T8'],
    'T7': ['F7', 'FC5', 'P7'],
    'T8': ['F8', 'FC6', 'P8'],
    'P7': ['T7'],
    'P8': ['T8'],
    'O1': ['O2'],
    'O2': ['O1']
}


# The following three lists of local-global-graph definitions are according the paper:  LGGNet
DREAMER_GENERAL_REGION_LIST = format_region_channel_list(DREAMER_CHANNEL_LIST,
                                                         GENERAL_REGION_LIST)
DREAMER_FRONTAL_REGION_LIST = format_region_channel_list(DREAMER_CHANNEL_LIST,
                                                         FRONTAL_REGION_LIST)
DREAMER_HEMISPHERE_REGION_LIST = format_region_channel_list(DREAMER_CHANNEL_LIST,
                                                            HEMISPHERE_REGION_LIST)

# The following one lists of local-global-graph definitions are according the paper:  GMSS
DREAMER_NEIGHBOR_REGION_LIST = format_region_channel_list(DREAMER_CHANNEL_LIST,
                                                            NEIGHBOR_REGION_LIST)

# The following one lists of local-global-graph definitions are according the paper: HDGNet
DREAMER_POSTERIOR_REGION_LIST = format_region_channel_list(DREAMER_CHANNEL_LIST,
                                                            POSTERIOR_REGION_LIST)

# the mask in HDGNet,limit the information propagation within the local-graph: include: GENERAL, FRONTAL, HEMISPHERE, NEIGHBOR, POSTERIOR

DREAMER_GENERAL_REGION_MASK_MATRIX = np.array(format_adj_matrix_from_adj_list(DREAMER_CHANNEL_LIST,
                                                              format_region_adjacent_list(DREAMER_CHANNEL_LIST,
                                                                                          GENERAL_REGION_LIST)))

DREAMER_FRONTAL_REGION_MASK_MATRIX = np.array(format_adj_matrix_from_adj_list(DREAMER_CHANNEL_LIST,
                                                              format_region_adjacent_list(DREAMER_CHANNEL_LIST,
                                                                                          FRONTAL_REGION_LIST)))

DREAMER_HEMISPHERE_REGION_MASK_MATRIX = np.array(format_adj_matrix_from_adj_list(DREAMER_CHANNEL_LIST,
                                                              format_region_adjacent_list(DREAMER_CHANNEL_LIST,
                                                                                          HEMISPHERE_REGION_LIST)))

DREAMER_NEIGHBOR_REGION_MASK_MATRIX = np.array(format_adj_matrix_from_adj_list(DREAMER_CHANNEL_LIST,
                                                              format_region_adjacent_list(DREAMER_CHANNEL_LIST,
                                                                                          NEIGHBOR_REGION_LIST)))

DREAMER_POSTERIOR_REGION_MASK_MATRIX = np.array(format_adj_matrix_from_adj_list(DREAMER_CHANNEL_LIST,
                                                              format_region_adjacent_list(DREAMER_CHANNEL_LIST,
                                                                                          POSTERIOR_REGION_LIST)))










