import numpy as np
from typing import List, Tuple, Dict


# get the adjacent list by local partition
def format_region_adjacent_list(channel_list, region_list):
    adjacent_list = {}
    for region in region_list:
        for region_channel in region:
            try:
                channel_list.index(region_channel)
            except:
                continue
            key = region_channel
            region_index_copy = region.copy()
            #region_index_copy.remove(region_channel)
            for value in region_index_copy:  # construct adjacent_list
                try:
                    channel_list.index(value)
                except:
                    continue
                if key in adjacent_list:
                    adjacent_list[key].append(value)
                else:
                    adjacent_list[key] = []
                    adjacent_list[key].append(value)
    return adjacent_list

# get the index for each channel by local partition
def format_region_channel_list(channel_list, region_list):
    output = []
    for region in region_list:
        region_channel_index_list = []
        for region_channel in region:
            try:
                channel_index = channel_list.index(region_channel)
            except:
                continue
            region_channel_index_list.append(channel_index)
        if len(region_channel_index_list) > 0:
            output.append(region_channel_index_list)
    return output


'''
    return a list which is adjacent matrix according to 'adj_list', if the location_ij value is 1, that means node_i connect node_j 
'''
def format_adj_matrix_from_adj_list(channel_list: List,
                                    adj_list: List) -> List[List]:
    node_map = {k: i for i, k in enumerate(channel_list)}
    adj_matrix = np.zeros((len(channel_list), len(channel_list)))

    for start_node_name in adj_list:
        if not start_node_name in channel_list:
            continue
        start_node_index = node_map[start_node_name]
        end_node_list = adj_list[start_node_name]

        for end_node_name in end_node_list:
            if not end_node_name in node_map:
                continue
            end_node_index = node_map[end_node_name]
            adj_matrix[start_node_index][end_node_index] = 1

    return adj_matrix.tolist()
