import pandas as pd
import numpy as np
import math
import random
import itertools
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns


def create_pool(size):
    '''
    Creating an initial population containing 4 different types of
    individuals:

    SL: Selfish and Large
    SS: Selfish and Small
    SM: Selfish and Medium
    CL: Cooperative and Large
    CS: Cooperative and Small
    CM: Cooperative and Medium
    '''
    res = np.repeat(['SL', 'SS', 'SM', 'CL', 'CS', 'CM'], size/6)
    random.shuffle(res)
    return res


def divide_in_groups(pool, large_g=40, small_g=4, medium_g=22):
    '''
    Dividing the current population into two main divisions: one containing
    all the different types of large individuals and one containing instead
    all the types of small individuals. These two divisions are additionally
    splitted in multiple groups depending on the large_g and small_g parameters
    which represent respectively the fixed group size that each large and small
    group should have. If there are not enough individuals left to fill a group
    they are automatically discarded.
    '''
    large = [ind for ind in pool if ind[1] == 'L']
    small = [ind for ind in pool if ind[1] == 'S']
    medium = [ind for ind in pool if ind[1] == 'M']
    discard_large = int(large_g*(len(large)/large_g - math.floor(len(large)/large_g)))
    discard_small = int(small_g*(len(small)/small_g - math.floor(len(small)/small_g)))
    discard_medium = int(medium_g*(len(medium)/medium_g - math.floor(len(medium)/medium_g)))
    try:
        groups_l = np.array(large[: len(large) - discard_large])
        groups_l = groups_l.reshape(math.floor(len(large)/large_g), -1)
    except:
        if groups_l.size == 0:
            groups_s = np.array(small[: len(small) - discard_small])
            groups_s = groups_s.reshape(math.floor(len(small)/small_g), -1)
            groups_m = np.array(medium[: len(medium) - discard_medium])
            if groups_m.size != 0 and groups_m.size != 1:
                groups_m = groups_m.reshape(math.floor(len(medium)/medium_g), -1)
            return groups_l, groups_s, groups_m
        groups_l = np.array(large[: len(large) - discard_large - 1])
        groups_l = groups_l.reshape(math.floor(len(large)/large_g), -1)
    try:
        groups_m = np.array(medium[: len(medium) - discard_medium])
        groups_m = groups_m.reshape(math.floor(len(medium)/medium_g), -1)
    except:
        if groups_m.size == 0:
            groups_s = np.array(small[: len(small) - discard_small])
            if groups_s.size != 0 and groups_s.size != 1:
                groups_s = groups_s.reshape(math.floor(len(small)/small_g), -1)
            return groups_l, groups_s, groups_m
        groups_m = np.array(medium[: len(medium) - discard_medium -1])
        groups_m = groups_m.reshape(math.floor(len(medium)/medium_g), -1)
    try:
        groups_s = np.array(small[: len(small) - discard_small])
        groups_s = groups_s.reshape(math.floor(len(small)/small_g), -1)
    except:
        if groups_s.size == 0:
            return groups_l, groups_s, groups_m
        groups_s = np.array(small[: len(small) - discard_small -1])
        groups_s = groups_s.reshape(math.floor(len(small)/small_g), -1)
    return groups_l, groups_s, groups_m


def reproduction(large_gs, small_gs, medium_gs, disposal_limit=4, large_r=50, small_r=4, medium_r=27, self_g=0.02, coop_g=0.018, self_c=0.2,
                 coop_c=0.1, K=0.1):
    '''
    Reproduction takes place just within divisions and they are dependent
    on the magnitude of the share of the total group resource that the
    genotype receives and the replicator equations (shown above).
    Therefore, the reproduction results are highly dependent of the disposal
    time and the equations parameters.
    '''
    i = 0
    large_g_res = [[]] * len(large_gs)
    small_g_res = [[]] * len(small_gs)
    medium_g_res = [[]] * len(medium_gs)
    for large_g, small_g, medium_g in itertools.zip_longest(large_gs, small_gs, medium_gs):
        if small_g is None and large_g is not None and medium_g is not None:
            unique, counts = np.unique(large_g, return_counts=True)
            large_counts = dict(zip(unique, counts))
            disp_time = 0
            large_coop_individuals = large_counts.get('CL', 0)
            large_self_individuals = large_counts.get('SL', 0)
            unique, counts = np.unique(medium_g, return_counts=True)
            medium_counts = dict(zip(unique, counts))
            medium_coop_individuals = medium_counts.get('CM', 0)
            medium_self_individuals = medium_counts.get('SM', 0)
            while disp_time != disposal_limit:
                large_coop_R_i = (large_coop_individuals * coop_g * coop_c) / (
                            large_coop_individuals * coop_g * coop_c + large_self_individuals * self_g * self_c) * large_r
                large_self_R_i = (large_self_individuals * self_g * self_c) / (
                            large_self_individuals * self_g * self_c + large_coop_individuals * coop_g * coop_c) * large_r
                large_coop_individuals = large_coop_individuals + large_coop_R_i / coop_c - K * large_coop_individuals
                large_self_individuals = large_self_individuals + large_self_R_i / self_c - K * large_self_individuals
                medium_coop_R_i = (medium_coop_individuals * coop_g * coop_c) / (
                            medium_coop_individuals * coop_g * coop_c + medium_self_individuals * self_g * self_c) * medium_r
                medium_self_R_i = (medium_self_individuals * self_g * self_c) / (
                            medium_self_individuals * self_g * self_c + medium_coop_individuals * coop_g * coop_c) * medium_r
                medium_coop_individuals = medium_coop_individuals + medium_coop_R_i / coop_c - K * medium_coop_individuals
                medium_self_individuals = medium_self_individuals + medium_self_R_i / self_c - K * medium_self_individuals
                disp_time += 1
            medium_g_res[i] = []
            medium_g_res[i].extend(['CM'] * int(medium_coop_individuals))
            medium_g_res[i].extend(['SM'] * int(medium_self_individuals))
            large_g_res[i] = []
            large_g_res[i].extend(['CL'] * int(large_coop_individuals))
            large_g_res[i].extend(['SL'] * int(large_self_individuals))
            i += 1
            continue
        if small_g is None and large_g is not None:
            unique, counts = np.unique(large_g, return_counts=True)
            large_counts = dict(zip(unique, counts))
            disp_time = 0
            large_coop_individuals = large_counts.get('CL', 0)
            large_self_individuals = large_counts.get('SL', 0)
            while disp_time != disposal_limit:
                large_coop_R_i = (large_coop_individuals * coop_g * coop_c) / (
                            large_coop_individuals * coop_g * coop_c + large_self_individuals * self_g * self_c) * large_r
                large_self_R_i = (large_self_individuals * self_g * self_c) / (
                            large_self_individuals * self_g * self_c + large_coop_individuals * coop_g * coop_c) * large_r
                large_coop_individuals = large_coop_individuals + large_coop_R_i / coop_c - K * large_coop_individuals
                large_self_individuals = large_self_individuals + large_self_R_i / self_c - K * large_self_individuals
                disp_time += 1
            large_g_res[i] = []
            large_g_res[i].extend(['CL'] * int(large_coop_individuals))
            large_g_res[i].extend(['SL'] * int(large_self_individuals))
            i += 1
            continue
        if small_g is None and medium_g is not None:
            unique, counts = np.unique(medium_g, return_counts=True)
            medium_counts = dict(zip(unique, counts))
            medium_coop_individuals = medium_counts.get('CM', 0)
            medium_self_individuals = medium_counts.get('SM', 0)
            while disp_time != disposal_limit:
                medium_coop_R_i = (medium_coop_individuals * coop_g * coop_c) / (
                            medium_coop_individuals * coop_g * coop_c + medium_self_individuals * self_g * self_c) * medium_r
                medium_self_R_i = (medium_self_individuals * self_g * self_c) / (
                            medium_self_individuals * self_g * self_c + medium_coop_individuals * coop_g * coop_c) * medium_r
                medium_coop_individuals = medium_coop_individuals + medium_coop_R_i / coop_c - K * medium_coop_individuals
                medium_self_individuals = medium_self_individuals + medium_self_R_i / self_c - K * medium_self_individuals
                disp_time += 1
            medium_g_res[i] = []
            medium_g_res[i].extend(['CM'] * int(medium_coop_individuals))
            medium_g_res[i].extend(['SM'] * int(medium_self_individuals))
            i += 1
            continue
        if large_g is None and medium_g is not None:
            unique, counts = np.unique(medium_g, return_counts=True)
            medium_counts = dict(zip(unique, counts))
            disp_time = 0
            medium_coop_individuals = medium_counts.get('CM', 0)
            medium_self_individuals = medium_counts.get('SM', 0)
            unique, counts = np.unique(small_g, return_counts=True)
            small_counts = dict(zip(unique, counts))
            small_coop_individuals = small_counts.get('CS', 0)
            small_self_individuals = small_counts.get('SS', 0)
            while disp_time != disposal_limit:
                medium_coop_R_i = (medium_coop_individuals * coop_g * coop_c) / (
                            medium_coop_individuals * coop_g * coop_c + medium_self_individuals * self_g * self_c) * medium_r
                medium_self_R_i = (medium_self_individuals * self_g * self_c) / (
                            medium_self_individuals * self_g * self_c + medium_coop_individuals * coop_g * coop_c) * medium_r
                medium_coop_individuals = medium_coop_individuals + medium_coop_R_i / coop_c - K * medium_coop_individuals
                medium_self_individuals = medium_self_individuals + medium_self_R_i / self_c - K * medium_self_individuals
                small_coop_R_i = (small_coop_individuals * coop_g * coop_c) / (
                            small_coop_individuals * coop_g * coop_c + small_self_individuals * self_g * self_c) * small_r
                small_self_R_i = (small_self_individuals * self_g * self_c) / (
                            small_self_individuals * self_g * self_c + small_coop_individuals * coop_g * coop_c) * small_r
                small_coop_individuals = small_coop_individuals + small_coop_R_i / coop_c - K * small_coop_individuals
                small_self_individuals = small_self_individuals + small_self_R_i / self_c - K * small_self_individuals
                disp_time += 1

            medium_g_res[i] = []
            medium_g_res[i].extend(['CM'] * int(medium_coop_individuals))
            medium_g_res[i].extend(['SM'] * int(medium_self_individuals))
            small_g_res[i] = []
            small_g_res[i].extend(['CS'] * int(small_coop_individuals))
            small_g_res[i].extend(['SS'] * int(small_self_individuals))
            i += 1
            continue
        if medium_g is None and large_g is None:
            unique, counts = np.unique(small_g, return_counts=True)
            small_counts = dict(zip(unique, counts))
            disp_time = 0
            small_coop_individuals = small_counts.get('CS', 0)
            small_self_individuals = small_counts.get('SS', 0)
            while disp_time != disposal_limit:
                small_coop_R_i = (small_coop_individuals * coop_g * coop_c) / (
                            small_coop_individuals * coop_g * coop_c + small_self_individuals * self_g * self_c) * small_r
                small_self_R_i = (small_self_individuals * self_g * self_c) / (
                            small_self_individuals * self_g * self_c + small_coop_individuals * coop_g * coop_c) * small_r
                small_coop_individuals = small_coop_individuals + small_coop_R_i / coop_c - K * small_coop_individuals
                small_self_individuals = small_self_individuals + small_self_R_i / self_c - K * small_self_individuals
                disp_time += 1

            small_g_res[i] = []
            small_g_res[i].extend(['CS'] * int(small_coop_individuals))
            small_g_res[i].extend(['SS'] * int(small_self_individuals))
            i += 1
            continue
        if medium_g is not None and large_g is not None:
            unique, counts = np.unique(large_g, return_counts=True)
            large_counts = dict(zip(unique, counts))
            disp_time = 0
            large_coop_individuals = large_counts.get('CL', 0)
            large_self_individuals = large_counts.get('SL', 0)
            unique, counts = np.unique(small_g, return_counts=True)
            small_counts = dict(zip(unique, counts))
            small_coop_individuals = small_counts.get('CS', 0)
            small_self_individuals = small_counts.get('SS', 0)
            unique, counts = np.unique(medium_g, return_counts=True)
            medium_counts = dict(zip(unique, counts))
            medium_coop_individuals = medium_counts.get('CM', 0)
            medium_self_individuals = medium_counts.get('SM', 0)
            while disp_time != disposal_limit:
                large_coop_R_i = (large_coop_individuals * coop_g * coop_c) / (
                            large_coop_individuals * coop_g * coop_c + large_self_individuals * self_g * self_c) * large_r
                large_self_R_i = (large_self_individuals * self_g * self_c) / (
                            large_self_individuals * self_g * self_c + large_coop_individuals * coop_g * coop_c) * large_r
                large_coop_individuals = large_coop_individuals + large_coop_R_i / coop_c - K * large_coop_individuals
                large_self_individuals = large_self_individuals + large_self_R_i / self_c - K * large_self_individuals
                small_coop_R_i = (small_coop_individuals * coop_g * coop_c) / (
                            small_coop_individuals * coop_g * coop_c + small_self_individuals * self_g * self_c) * small_r
                small_self_R_i = (small_self_individuals * self_g * self_c) / (
                            small_self_individuals * self_g * self_c + small_coop_individuals * coop_g * coop_c) * small_r
                small_coop_individuals = small_coop_individuals + small_coop_R_i / coop_c - K * small_coop_individuals
                small_self_individuals = small_self_individuals + small_self_R_i / self_c - K * small_self_individuals
                medium_coop_R_i = (medium_coop_individuals * coop_g * coop_c) / (
                            medium_coop_individuals * coop_g * coop_c + medium_self_individuals * self_g * self_c) * medium_r
                medium_self_R_i = (medium_self_individuals * self_g * self_c) / (
                            medium_self_individuals * self_g * self_c + medium_coop_individuals * coop_g * coop_c) * medium_r
                medium_coop_individuals = medium_coop_individuals + medium_coop_R_i / coop_c - K * medium_coop_individuals
                medium_self_individuals = medium_self_individuals + medium_self_R_i / self_c - K * medium_self_individuals
                disp_time += 1

            large_g_res[i] = []
            large_g_res[i].extend(['CL'] * int(large_coop_individuals))
            large_g_res[i].extend(['SL'] * int(large_self_individuals))
            small_g_res[i] = []
            small_g_res[i].extend(['CS'] * int(small_coop_individuals))
            small_g_res[i].extend(['SS'] * int(small_self_individuals))
            medium_g_res[i] = []
            medium_g_res[i].extend(['CM'] * int(medium_coop_individuals))
            medium_g_res[i].extend(['SM'] * int(medium_self_individuals))
            i += 1
            continue

    return large_g_res, small_g_res, medium_g_res


def update_pool(large_gs, small_gs, medium_gs, mutation=False):
    '''
    In this case, are taken as input the two divisions which had undergone
    reproduction and are merged together to create a pool like the one created
    in the initialization step. In this situation, we need to make sure that
    overall population size remains the same.
    '''
    res = reduce(lambda x,y :x+y ,large_gs, []) + reduce(lambda x,y :x+y , small_gs, []) + reduce(lambda x,y :x+y , medium_gs, [])
    unique, counts = np.unique(res, return_counts=True)
    pop_counts = dict(zip(unique, counts))
    new_pop_elements = [int(((i/len(res))*pop)) for i in pop_counts.values()]
    res = np.repeat([*pop_counts.keys()], new_pop_elements).tolist()
    random.shuffle(res)
    if mutation is True:
        for i in range(0, int(len(res)/2)):
            res[i] = (np.random.choice(['C' + res[i][1], 'S' + res[i][1]],
                            p=[1/2, 1/2]))
    return res


pop = 4000
iter_num = 150
d = {'Iteration Num': [0], 'Selfish and Large': [666], 'Selfish and Small': [666], 'Selfish and Medium': [666],
     'Cooperative and Large': [666], 'Cooperative and Small': [666], 'Cooperative and Medium': [666]}
df = pd.DataFrame(data=d)
migrant_pool = create_pool(pop)
for i in range(1, iter_num + 1):
    large_group, small_group, medium_group = divide_in_groups(migrant_pool)
    large_group, small_group, medium_group = reproduction(large_group, small_group, medium_group)
    migrant_pool = update_pool(large_group, small_group, medium_group)
    unique, counts = np.unique(migrant_pool, return_counts=True)
    res_counts = dict(zip(unique, counts))
    df = df.append({"Iteration Num": i,
                    "Selfish and Large": res_counts.get('SL', 0),
                    "Selfish and Small": res_counts.get('SS', 0),
                    "Selfish and Medium": res_counts.get('SM', 0),
                    "Cooperative and Large": res_counts.get('CL', 0),
                    "Cooperative and Small": res_counts.get('CS', 0),
                    "Cooperative and Medium": res_counts.get('CM', 0)
                    }, ignore_index=True)
    if i % 50 == 0:
        print('Iteration Number:', i)