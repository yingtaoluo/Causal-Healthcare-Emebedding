import numpy as np
import csv
import torch
import pdb


def grouping_dia_icd(dia_list):
    # construct two lists of group and icds
    new, group_code, group_icds, flag = [], [], [], 0  # flag 0 denotes first line
    with open('./data/AppendixASingleDX.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i < 4 or line.strip('\n') == '':
                flag = 0
                continue
            else:
                if flag == 0:
                    group_code.append([x for x in line.strip('\n').split(' ') if x != ''][0])
                    if len(new) != 0:
                        group_icds.append(new)
                    new = []
                    flag = 1
                else:
                    new += [x for x in line.strip('\n').split(' ') if x != '']
        group_icds.append(new)

    # convert each item from icd to group
    for i, patient in enumerate(dia_list):
        for j, visit in enumerate(patient):
            for k, icd in enumerate(visit):
                # replace icd with group code
                for r, group_icd in enumerate(group_icds):
                    if icd in group_icd:
                        dia_list[i][j][k] = group_code[r]
            dia_list[i][j] = list(set(dia_list[i][j]))  # remove repeated elements

    return dia_list


def grouping_pro_icd(pro_list):
    # construct two lists of group and icds
    new, group_code, group_icds, flag = [], [], [], 0  # flag 0 denotes first line
    with open('./data/AppendixBSinglePR.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i < 4 or line.strip('\n') == '':
                flag = 0
                continue
            else:
                if flag == 0:
                    group_code.append([x for x in line.strip('\n').split(' ') if x != ''][0])
                    if len(new) != 0:
                        group_icds.append(new)
                    new = []
                    flag = 1
                else:
                    new += [x for x in line.strip('\n').split(' ') if x != '']
        group_icds.append(new)

    # convert each item from icd to group
    for i, patient in enumerate(pro_list):
        for j, visit in enumerate(patient):
            for k, icd in enumerate(visit):
                # replace icd with group code
                for r, group_icd in enumerate(group_icds):
                    if icd in group_icd:
                        pro_list[i][j][k] = group_code[r]
            pro_list[i][j] = list(set(pro_list[i][j]))  # remove repeated elements

    return pro_list

