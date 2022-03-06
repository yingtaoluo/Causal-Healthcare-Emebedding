import pdb
import csv
import copy
import numpy as np
import torch
import torch.nn.functional as func


acc_rank = [10, 20]


def grouping_dia_icd(dia_list):
    # construct two lists of group and icds
    group_code, group_icds = [], []
    with open('./data/dxref.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        sheet = np.array([x for x in np.array(rows[3:])])[:, :2]

    for item in sheet:
        if eval(item[1]).strip(' ') not in group_code:
            group_code.append(eval(item[1]).strip(' '))
            group_icds.append([])
            group_icds[-1].append(eval(item[0]).strip(' '))
        else:
            group_icds[-1].append(eval(item[0]).strip(' '))

    # convert each item from icd to group
    for i, patient in enumerate(dia_list):
        for j, visit in enumerate(patient):
            for k, icd in enumerate(visit):
                # replace icd with group code
                for r, group_icd in enumerate(group_icds):
                    if icd in group_icd:
                        dia_list[i][j][k] = group_code[r]
                        break
                if dia_list[i][j][k] not in group_code:
                    dia_list[i][j][k] = ''

            dia_list[i][j] = list(filter(None, dia_list[i][j]))
            dia_list[i][j] = list(set(dia_list[i][j]))  # remove repeated elements

    return dia_list


def grouping_pro_icd(pro_list):
    # construct two lists of group and icds
    group_code, group_icds = [], []
    with open('./data/prref.csv', "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        sheet = np.array([x for x in np.array(rows[3:])])[:, :2]

    for item in sheet:
        if eval(item[1]).strip(' ') not in group_code:
            group_code.append(eval(item[1]).strip(' '))
            group_icds.append([])
            group_icds[-1].append(eval(item[0]).strip(' '))
        else:
            group_icds[-1].append(eval(item[0]).strip(' '))

    # convert each item from icd to group
    for i, patient in enumerate(pro_list):
        for j, visit in enumerate(patient):
            for k, icd in enumerate(visit):
                # replace icd with group code
                for r, group_icd in enumerate(group_icds):
                    if icd in group_icd:
                        pro_list[i][j][k] = group_code[r]
                        break
                if pro_list[i][j][k] not in group_code:
                    pro_list[i][j][k] = ''

            pro_list[i][j] = list(filter(None, pro_list[i][j]))
            pro_list[i][j] = list(set(pro_list[i][j]))  # remove repeated elements

    return pro_list


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

    return max_len_seq-1, max_size_basket, dia_icds, pro_icds  # max_len_seq needs to minus 1 for both data and label


def divide_data_label(seq1, seq2):
    data_seq1 = copy.deepcopy(seq1)
    data_seq2 = copy.deepcopy(seq2)
    label_seq1 = copy.deepcopy(seq1)
    real_len = []  # seq len of each subject
    for i, subject in enumerate(seq1):
        data_seq1[i], data_seq2[i], label_seq1[i] = [], [], []
        count = 0
        for j, basket in enumerate(subject):
            count += 1
            if j == 0:
                data_seq1[i].append(seq1[i][j])
                data_seq2[i].append(seq2[i][j])
            elif j == len(subject)-1:
                label_seq1[i].append(seq1[i][j])
            else:
                data_seq1[i].append(seq1[i][j])
                data_seq2[i].append(seq2[i][j])
                label_seq1[i].append(seq1[i][j])
        real_len.append(count-1)  # real len for both data and label should minus 1

    return data_seq1, data_seq2, label_seq1, real_len


def convert_one_hot(sequence, size_icd):
    for i, subject in enumerate(sequence):
        for j, basket in enumerate(subject):
            sequence[i][j] = torch.sum(func.one_hot(torch.tensor(basket).to(torch.int64), num_classes=size_icd), dim=0)
        sequence[i] = torch.tensor([item.numpy() for item in sequence[i]])

    sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True)
    print(sequence.shape)

    return sequence


def to_npy(x):
    return x.cpu().data.numpy() if torch.cuda.is_available() else x.detach().numpy()


def construct_data_combo(dia, pro, label, length, ratio):
    dia = dia.to(torch.float32)
    pro = pro.to(torch.float32)
    label = label.to(torch.float32)
    length = torch.tensor(length)

    # randomly divide train/dev/test datasets
    ix = np.linspace(0, len(dia)-1, len(dia))
    np.random.seed(0)  # fix data divide seed if needed
    np.random.shuffle(ix)
    train_ratio, dev_ratio, test_ratio = ratio
    train_end, dev_end = int(train_ratio*len(dia)), int((train_ratio+dev_ratio)*len(dia))
    test_end = int((train_ratio+dev_ratio+test_ratio)*len(dia))

    train_dia = dia[ix[:train_end]]
    valid_dia = dia[ix[train_end:dev_end]]
    test_dia = dia[ix[dev_end:test_end]]
    train_pro = pro[ix[:train_end]]
    valid_pro = pro[ix[train_end:dev_end]]
    test_pro = pro[ix[dev_end:test_end]]
    train_label = label[ix[:train_end]]
    valid_label = label[ix[train_end:dev_end]]
    test_label = label[ix[dev_end:test_end]]
    train_length = length[ix[:train_end]]
    valid_length = length[ix[train_end:dev_end]]
    test_length = length[ix[dev_end:test_end]]

    combo = [train_dia, valid_dia, test_dia, train_pro, valid_pro, test_pro,
             train_label, valid_label, test_label, train_length, valid_length, test_length]

    return combo


def calculate_ndcg(batch_prob, batch_label, batch_len):
    # prob (N, T, U) --> (*, U), label (N, T, U) --> (*, U), length (N)
    prob, label = [], []
    for i in range(batch_prob.shape[0]):
        for j in range(batch_len[i]):
            prob.append(to_npy(batch_prob[i, j]))
            label.append(to_npy(batch_label[i, j]))
    scores, labels = torch.tensor(prob), torch.tensor(label, dtype=torch.int)

    rank = (-scores).argsort(dim=-1)

    ndcg = np.zeros((len(acc_rank), ))
    for i, k in enumerate(acc_rank):
        cut = rank[:, :k]
        hits = labels.gather(1, cut)  # the label values of the top scores
        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits.float() * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
        ndcg[i] = (dcg / idcg).sum().item()

    return ndcg


def calculate_acc(batch_prob, batch_label, length):
    # prob (N, T, U) --> (*, U), label (N, T, U) --> (*, U), length (N)
    prob, label = [], []
    for i in range(batch_prob.shape[0]):
        for j in range(length[i]):
            prob.append(to_npy(batch_prob[i, j]))
            label.append(to_npy(batch_label[i, j]))
    prob, label = torch.tensor(prob), torch.tensor(label)

    acc = np.zeros((len(acc_rank), ))
    for i, k in enumerate(acc_rank):
        # top_k_batch (*, k), top_k (k)
        _, top_k_batch = torch.topk(prob, k=k)
        for j, top_k in enumerate(top_k_batch):
            label_pool = torch.nonzero(label[j])[:, 0]
            for top_each in top_k:
                if top_each in label_pool:
                    acc[i] += 1/len(label_pool)

    return acc


# not used
def calculate_group_acc(batch_prob, batch_label, length, groups):
    # prob (N, T, U) --> (*, U), label (N, T, U) --> (*, U), length (N)
    prob, label = [], []
    for i in range(batch_prob.shape[0]):
        for j in range(length[i]):
            prob.append(to_npy(batch_prob[i, j]))
            label.append(to_npy(batch_label[i, j]))
    prob, label = torch.tensor(prob), torch.tensor(label)

    hit = np.zeros((len(acc_rank), len(groups)))  # first dimension depends on @
    num = np.zeros((len(groups)))  # number of icd appearance for each group

    for i, k in enumerate(acc_rank):
        # top_k_batch (*, k), top_k (k)
        _, top_k_batch = torch.topk(prob, k=k)
        for j, top_k in enumerate(top_k_batch):
            label_pool = torch.nonzero(label[j])[:, 0]
            for label_each in label_pool:
                # know which group this label is in
                g = 0
                while label_each not in groups[g]:
                    g += 1

                # add to 'num' and 'hit'
                if i == 0:  # need not to count after first iteration
                    num[g] += 1

                if label_each in top_k:  # only count if is in topk
                    hit[i, g] += 1

    return hit, num

