'''add the root path here'''
import sys
sys.path.append(r"/home/luoyingtao/Causal")

from utils import convert_one_hot, divide_data_label
import numpy as np
import copy
import torch
import csv
import joblib
import pdb


def discover_properties(seq1, seq2):
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

    pdb.set_trace()

    return max_len_seq-1, max_size_basket, dia_icds, pro_icds  # max_len_seq needs to minus 1 for both data and label


def form_seq(dia, pro, admission, max_len):  # from csv_shape to sequence_shape
    ''' Assume seq_num is already ordered in the original dataset '''
    # reorder the sequence of a patient w.r.t. the medical visit time
    def find_temporal_order(ids, adt):
        time_seq = []
        for ID in ids:
            ix = np.argwhere(adt[:, 2] == ID)[0][0]
            # turn year-month-day into an integer to approximately estimate their order chronologically
            time_seq.append(int(adt[ix, 3][0:4] + adt[ix, 3][5:7] + adt[ix, 3][8:10]))
        indexes = np.argsort(time_seq)

        return indexes.tolist()

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

                last_hadm = row[2]
                hadm_seq = [row[-1]]  # reset and record present row

        else:  # record seq before entering next subject
            last_subject = row[1]
            last_hadm = row[2]
            dia_sub_seq.append(hadm_seq)
            if len(dia_sub_seq) >= max_len:  # filter out subjects that have too few visits
                temporal_index = find_temporal_order(hadm_ids, admission)  # correct index order w.r.t. time
                dia_seq.append([x for _, x in sorted(zip(temporal_index, dia_sub_seq))])
                pro_seq.append([x for _, x in sorted(zip(temporal_index, pro_sub_seq))])

            dia_sub_seq = []
            pro_sub_seq = []

            hadm_seq = [row[-1]]  # reset and record the present row
            hadm_ids = []  # reset

    return dia_seq, pro_seq


if __name__ == "__main__":
    with open('./data/mimic3/DIAGNOSES_ICD.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        mimic3_dia = np.array(rows[1:])
        print('{} rows in total'.format(len(mimic3_dia)))

    with open('./data/mimic3/PROCEDURES_ICD.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        mimic3_pro = np.array(rows[1:])
        print('{} rows in total'.format(len(mimic3_pro)))

    with open('./data/mimic3/ADMISSIONS.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        mimic3_admit = np.array(rows[1:])[:, :4]
        print('{} rows in total'.format(len(mimic3_admit)))

    dia_list, pro_list = form_seq(mimic3_dia, mimic3_pro, mimic3_admit, 3)  # w.r.t. patient, time_order, basket_order

    all_visits = 0
    all_codes = 0
    for patient in dia_list:
        all_visits += len(patient)
        for visit in patient:
            all_codes += len(visit)

    print('Total num of visits:{}'.format(all_visits))
    print('Total num of patients:{}'.format(len(dia_list)))
    print('Avg num of visits:{}'.format(all_visits/len(dia_list)))
    print('Avg num of codes in a visit:{}'.format(all_codes/all_visits))

    # dia_list, pro_list = grouping_dia_icd(dia_list), grouping_pro_icd(pro_list)  # no need for small dataset
    len_seq, size_basket, unique_icds, pro_unique_icds = discover_properties(dia_list, pro_list)

    pdb.set_trace()

    print('Total num of unique ICD codes (diagnoses):{}'.format(len(unique_icds)))
    print('Total num of unique ICD codes (procedures):{}'.format(len(pro_unique_icds)))

    data_dia, data_pro, label_dia, true_len = divide_data_label(dia_list, pro_list)

    # (N, T, U)
    data_dia = convert_one_hot(data_dia, len(unique_icds))
    data_pro = convert_one_hot(data_pro, len(unique_icds))
    label_dia = convert_one_hot(label_dia, len(unique_icds))

    data_pkl = './data/mimic3/dataset.pkl'
    open(data_pkl, 'a')
    with open(data_pkl, 'wb') as pkl:
        joblib.dump([data_dia, data_pro, label_dia, true_len], pkl)

