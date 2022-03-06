'''add the root path here'''
import sys
sys.path.append(r"/home/luoyingtao/Causal")

from utils import convert_one_hot, divide_data_label, grouping_dia_icd, grouping_pro_icd
import numpy as np
import copy
import torch
import csv
import joblib
import pdb


# python data/mimic3/preprocess_insurance.py
def form_seq(dia, pro, admission, max_len, ins):  # from csv_shape to sequence_shape
    ''' Assume seq_num is already ordered in the original dataset '''
    # reorder the sequence of a patient w.r.t. the medical visit time
    def find_temporal_order(ids, adt):
        options = [False]*len(ids)
        time_seq = []
        for i, ID in enumerate(ids):
            ix = np.argwhere(adt[:, 2] == ID)[0][0]
            # turn year-month-day into an integer to approximately estimate their order chronologically
            if adt[ix, 9] == ins:
                st = adt[ix, 3].split('/')
                # time_seq.append(float(adt[ix, 3][0:4] + adt[ix, 3][5:7] + adt[ix, 3][8:10]))
                time_seq.append(int(st[0])*12 + int(st[1]) + int(st[2][0:2])/12)  # approximate
                options[i] = True
        indexes = np.argsort(time_seq)

        return indexes.tolist(), options

    last_subject = dia[0, 1]
    last_hadm = dia[0, 2]
    dia_seq, dia_sub_seq = [], []
    pro_seq, pro_sub_seq = [], []
    hadm_seq = []  # record the icds of a hadm
    hadm_ids = []  # reserve the ids to look up the time and reorder

    for row in dia:
        if row[1] == last_subject:  # still the subject
            if row[2] not in hadm_ids:  # get unique hadm ids of the subject
                hadm_ids.append(row[2])

            if row[2] == last_hadm:  # still the same hadm then append the icd
                hadm_seq.append(row[-1])

            else:  # record seq before entering next hadm
                is_lonely_hadm = True if last_hadm not in pro[:, 2] else False
                if is_lonely_hadm is False:  # only add hadm_seq if this hadm also exists in pro
                    dia_sub_seq.append(hadm_seq)
                    hadm_ixs = np.argwhere(pro[:, 2] == last_hadm)[:, 0]  # [?, 1] --> [?]
                    pro_sub_seq.append(pro[hadm_ixs, -1].tolist())
                else:  # added after detecting a bug of length mismatch between hadm_ids and dia_sub
                    if last_hadm not in hadm_ids:
                        pdb.set_trace()
                    hadm_ids.remove(last_hadm)

                last_hadm = row[2]
                hadm_seq = [row[-1]]  # reset and record present row

        else:  # record seq before entering next subject
            last_subject = row[1]
            last_hadm = row[2]
            dia_sub_seq.append(hadm_seq)  # no need to record pro_sub_seq
            if len(dia_sub_seq) >= max_len:  # filter out subjects that have too few visits
                temporal_index, opts = find_temporal_order(hadm_ids, admission)  # correct index order w.r.t. time
                dia_sub_tmp, pro_sub_tmp = [], []

                for ix, opt in enumerate(opts):
                    if opt is True:
                        dia_sub_tmp.append(dia_sub_seq[ix])
                        if ix < len(pro_sub_seq):
                            pro_sub_tmp.append(pro_sub_seq[ix])

                if len(dia_sub_tmp) >= max_len:
                    dia_seq.append([x for _, x in sorted(zip(temporal_index, dia_sub_tmp))])
                    pro_seq.append([x for _, x in sorted(zip(temporal_index, pro_sub_tmp))])

            dia_sub_seq = []
            pro_sub_seq = []

            hadm_seq = [row[-1]]  # reset and record the present row
            hadm_ids = [row[2]]  # reset  # modify to [row[2]] instead of [] after detecting the bug

    return dia_seq, pro_seq


def properties(seq1, seq2, seq3, seq4):
    max_size_basket = 0
    max_len_seq = 0
    dia_icds, pro_icds = [], []  # convert icds into indexes for easier embedding
    for i, subject in enumerate(seq1):
        if len(subject) > max_len_seq:
            max_len_seq = len(subject)
        for j, basket in enumerate(subject):
            if len(basket) > max_size_basket:
                max_size_basket = len(basket)
            for k, icd in enumerate(basket):
                if icd not in dia_icds:
                    dia_icds.append(icd)
                seq1[i][j][k] = dia_icds.index(icd)  # from str icd to int num (start from 1) if padding

    for i, subject in enumerate(seq2):
        if len(subject) > max_len_seq:
            max_len_seq = len(subject)
        for j, basket in enumerate(subject):
            if len(basket) > max_size_basket:
                max_size_basket = len(basket)
            for k, icd in enumerate(basket):
                if icd not in pro_icds:
                    pro_icds.append(icd)
                seq2[i][j][k] = pro_icds.index(icd)  # from str icd to int num (start from 1) if padding

    for i, subject in enumerate(seq3):
        if len(subject) > max_len_seq:
            max_len_seq = len(subject)
        for j, basket in enumerate(subject):
            if len(basket) > max_size_basket:
                max_size_basket = len(basket)
            for k, icd in enumerate(basket):
                if icd not in dia_icds:
                    dia_icds.append(icd)
                seq3[i][j][k] = dia_icds.index(icd)  # from str icd to int num (start from 1) if padding

    for i, subject in enumerate(seq4):
        if len(subject) > max_len_seq:
            max_len_seq = len(subject)
        for j, basket in enumerate(subject):
            if len(basket) > max_size_basket:
                max_size_basket = len(basket)
            for k, icd in enumerate(basket):
                if icd not in pro_icds:
                    pro_icds.append(icd)
                seq4[i][j][k] = pro_icds.index(icd)  # from str icd to int num (start from 1) if padding

    return max_len_seq-1, max_size_basket, dia_icds, pro_icds  # max_len_seq needs to minus 1 for both data and label


if __name__ == "__main__":
    insurance1 = 'Medicare'
    insurance2 = 'Private'
    with open('./data/mimic3/DIAGNOSES_ICD.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        mimic3_dia = np.array(rows[1:])
        print('{} rows in total'.format(len(mimic3_dia)))
        # np.argwhere(mimic3_pro[:,2]=='105083')

    with open('./data/mimic3/PROCEDURES_ICD.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        mimic3_pro = np.array(rows[1:])
        print('{} rows in total'.format(len(mimic3_pro)))

    with open('./data/mimic3/ADMISSIONS.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        mimic3_admit = np.array(rows[1:])
        print('{} rows in total'.format(len(mimic3_admit)))

    dia_list1, pro_list1 = form_seq(mimic3_dia, mimic3_pro, mimic3_admit, 3, insurance1)
    dia_list2, pro_list2 = form_seq(mimic3_dia, mimic3_pro, mimic3_admit, 3, insurance2)
    # w.r.t. patient, time_order, basket_order

    # dia_list, pro_list = grouping_dia_icd(dia_list), grouping_pro_icd(pro_list)  # no need for small dataset
    len_seq, size_basket, unique_icds, pro_unique_icds = properties(dia_list1, pro_list1, dia_list2, pro_list2)

    # pdb.set_trace()
    #
    # vis_dia, vis_pro = [], []
    # for i, subject in enumerate(dia_list1):
    #     for j, basket in enumerate(subject):
    #         if 154 in basket and 155 in basket:
    #             vis_dia.append(subject)
    #             vis_pro.append(pro_list1[i])
    #             print(i, j)
    #             print(basket)
    #             break
    #
    # pdb.set_trace()
    # [(i, item) for (i, item) in enumerate(vis_dia) if len(item[2])==4]

    print('Total num of unique ICD codes (diagnoses):{}'.format(len(unique_icds)))
    print('Total num of unique ICD codes (procedures):{}'.format(len(pro_unique_icds)))

    # first
    data_dia, data_pro, label_dia, true_len = divide_data_label(dia_list1, pro_list2)
    # data_dia, data_pro, label_dia, true_len = divide_data_label(vis_dia, vis_pro)

    # (N, T, U)
    data_dia = convert_one_hot(data_dia, len(unique_icds))
    data_pro = convert_one_hot(data_pro, len(unique_icds))
    label_dia = convert_one_hot(label_dia, len(unique_icds))

    data_pkl = './data/mimic3/dataset_' + insurance1 + '.pkl'
    open(data_pkl, 'a')
    with open(data_pkl, 'wb') as pkl:
        joblib.dump([data_dia, data_pro, label_dia, true_len], pkl)

    # [i for i,x in enumerate([icd[0:3] for icd in unique_icds]) if x == '250']
    # [i for i,x in enumerate([icd[0:3] for icd in unique_icds]) if x == '362']

    print('Total num of unique ICD codes (diagnoses):{}'.format(len(unique_icds)))
    print('Total num of unique ICD codes (procedures):{}'.format(len(pro_unique_icds)))

    # first
    data_dia, data_pro, label_dia, true_len = divide_data_label(dia_list1, pro_list1)

    # (N, T, U)
    data_dia = convert_one_hot(data_dia, len(unique_icds))
    data_pro = convert_one_hot(data_pro, len(unique_icds))
    label_dia = convert_one_hot(label_dia, len(unique_icds))

    data_pkl = './data/mimic3/dataset_' + insurance1 + '.pkl'
    open(data_pkl, 'a')
    with open(data_pkl, 'wb') as pkl:
        joblib.dump([data_dia, data_pro, label_dia, true_len], pkl)

    # second
    data_dia, data_pro, label_dia, true_len = divide_data_label(dia_list2, pro_list2)

    # (N, T, U)
    data_dia = convert_one_hot(data_dia, len(unique_icds))
    data_pro = convert_one_hot(data_pro, len(unique_icds))
    label_dia = convert_one_hot(label_dia, len(unique_icds))

    data_pkl = './data/mimic3/dataset_' + insurance2 + '.pkl'
    open(data_pkl, 'a')
    with open(data_pkl, 'wb') as pkl:
        joblib.dump([data_dia, data_pro, label_dia, true_len], pkl)

