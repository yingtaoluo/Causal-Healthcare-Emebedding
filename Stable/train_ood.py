'''add the root path here'''
import sys
sys.path.append(r"/home/luoyingtao/Causal")

'''add packages here'''
import time
import joblib
import argparse
from statistics import mean
from torch import optim
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import *

# N: batch_size, U: number of unique ICDs, T: number of timestamps, H: embedding dimension, B: size of bucket
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='choose which model to train')
parser.add_argument('seed', type=int, help='choose which seed to load')
parser.add_argument('gpu', type=int, help='choose which gpu device to use')
parser.add_argument('lr', type=float, help='learning rate for main loss')
parser.add_argument('data', type=str, help='choose a dataset, MIMIC3 or CMS')

aux_args = parser.add_argument_group('auxiliary')
aux_args.add_argument('--coef', type=float, help='coefficient to balance the two losses')
aux_args.add_argument('--batch_size', type=int, help='choose a batch size')
aux_args.add_argument('--hidden_size', type=int, help='choose a hidden size')
aux_args.add_argument('--drop', type=float, help='choose a drop out rate')
aux_args.add_argument('--reg', type=float, help='choose a regularization coefficient')
parser.set_defaults(coef=1e-1, batch_size=64, hidden_size=16, drop=0, reg=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
GPU = args.gpu
device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
# print('train_device:{}'.format(device))
# print('train_gpu:{}'.format(GPU))


def calculate_hsic(dia_feat, pro_feat, weight, gpu):
    # [N, T, H] --> [H, N*T]
    batch_size, time_size, feat_size = dia_feat.shape
    dia_feat = dia_feat.reshape(batch_size * time_size, -1).transpose(1, 0)
    pro_feat = pro_feat.reshape(batch_size * time_size, -1).transpose(1, 0)

    # weighting
    weight = weight.reshape(batch_size * time_size, -1).transpose(1, 0)
    dia_feat *= weight
    pro_feat *= weight

    kx = torch.unsqueeze(dia_feat, 1) - torch.unsqueeze(dia_feat, 2)
    Kx = torch.exp(- kx ** 2)
    ky = torch.unsqueeze(pro_feat, 1) - torch.unsqueeze(pro_feat, 2)
    Ky = torch.exp(- ky ** 2)

    hsic = torch.zeros((feat_size, feat_size)).to(gpu)

    for j, item1 in enumerate(Kx):
        for k, item2 in enumerate(Ky):
            Kxy = torch.matmul(item1, item2)  # [N*T, N*T]
            n = Kxy.shape[0]
            h = torch.trace(Kxy) / n ** 2 + torch.mean(item1) * torch.mean(item2) - 2 * torch.mean(Kxy) / n
            hsic[j, k] = h * n ** 2 / (n - 1) ** 2

    return hsic


def calculate_hsic2(dia_feat, pro_feat, weight, gpu):
    # [N, T, H] --> [N*T, H]
    batch_size, time_size, feat_size = dia_feat.shape
    dia_feat = dia_feat.reshape(batch_size * time_size, -1)
    pro_feat = pro_feat.reshape(batch_size * time_size, -1)

    # weighting
    weight = weight.reshape(batch_size * time_size, -1)
    dia_feat *= weight
    pro_feat *= weight

    kx = torch.unsqueeze(dia_feat, 1) - torch.unsqueeze(dia_feat, 2)
    Kx = torch.exp(- kx ** 2)
    ky = torch.unsqueeze(pro_feat, 1) - torch.unsqueeze(pro_feat, 2)
    Ky = torch.exp(- ky ** 2)

    Kxy = torch.matmul(Kx, Ky)
    hsic = torch.zeros((batch_size * time_size, )).to(gpu)
    for i, matrix in enumerate(Kxy):
        n = matrix.shape[0]
        h = torch.trace(matrix) / n ** 2 + torch.mean(Kx[i]) * torch.mean(Ky[i]) - 2 * torch.mean(matrix) / n
        hsic[i] = h * n**2 / (n - 1)**2

    return hsic.reshape(batch_size, time_size)


class DataSet(data.Dataset):
    def __init__(self, input1, input2, labels, length, weight):
        # (NUM, T, U), (NUM, T, U), (NUM)
        self.input1, self.input2, self.labels, self.length, self.weight = input1, input2, labels, length, weight

    def __getitem__(self, index):
        input1 = self.input1[index].to(device)
        input2 = self.input2[index].to(device)
        labels = self.labels[index].to(device)
        length = self.length[index].to(device)
        weight = self.weight[index].to(device)
        return input1, input2, labels, length, weight

    def __len__(self):
        return len(self.input1)


class Trainer:
    def __init__(self, m, all_data, record):
        # load other parameters
        self.model = m.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.num_epoch = 1000
        self.early_max = 20
        self.threshold = np.mean(record['ndcg_valid'][-1]) if load else 0  # 0 if not update

        [self.train_dia, self.valid_dia, self.test_dia,
         self.train_pro, self.valid_pro, self.test_pro,
         self.train_label, self.valid_label, self.test_label,
         self.train_len, self.valid_len, self.test_len] = all_data

        self.weight = torch.nn.Parameter(torch.ones(size=(self.train_dia.shape[0], self.train_dia.shape[1])))

        max_times = int(len(self.train_dia) / self.batch_size) * self.batch_size
        self.train_dia, self.train_label = self.train_dia[0:max_times], self.train_label[0:max_times]

        file = open('./data/' + args.data + '/dataset_vis.pkl', 'rb')
        file_data = joblib.load(file)
        [self.train_dia, self.train_pro, self.train_label, self.train_len] = file_data  # tensor (NUM, T, U)
        self.train_dia = self.train_dia.to(torch.float32)
        self.train_pro = self.train_pro.to(torch.float32)
        self.train_label = self.train_label.to(torch.float32)
        self.train_len = torch.tensor(self.train_len)

        self.train_dataset = DataSet(self.train_dia, self.train_pro, self.train_label, self.train_len, self.weight)
        self.valid_dataset = DataSet(self.valid_dia, self.valid_pro, self.valid_label, self.valid_len, self.weight)
        self.test_dataset = DataSet(self.test_dia, self.test_pro, self.test_label, self.test_len, self.weight)
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = data.DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=args.reg)
        cor_opt = optim.Adam([self.weight], lr=self.learning_rate*args.coef, weight_decay=0)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        init_ndcg, best_acc, best_ndcg = 0, 0, 0
        epoch = 0

        for t in range(self.num_epoch):
            print('{} epoch ...'.format(t))
            # settings or validation and test
            train_size, valid_size, test_size, loss_sum = 0, 0, 0, 0
            acc_valid, acc_test, ndcg_valid, ndcg_test = init_ndcg, init_ndcg, init_ndcg, init_ndcg

            # training
            print('training ...')
            self.model.train()
            corr = []
            bar = tqdm(total=self.train_dia.shape[0])
            for step, item in enumerate(self.train_loader):
                # get batch data, (N, T, U), (N, T, U), (N)
                batch_dia, batch_pro, batch_label, batch_len, batch_weight = item
                batch_prob_dia, batch_feature_dia = self.model(batch_dia, 'dia')
                batch_prob_pro, batch_feature_pro = self.model(batch_pro, 'pro')

                # train model parameters
                batch_prob = batch_prob_dia + batch_prob_pro  # (N, T, U)

                # olist = []
                # for jj, basket in enumerate(batch_label[2]):
                #     slist = []
                #     for ii, one_hot in enumerate(basket):
                #         if one_hot == 1:
                #             slist.append(ii)
                #     olist.append(slist)

                # for ii, subject in enumerate(batch_label):
                #     for jj, basket in enumerate(subject):
                #         if basket[155] == 1:
                #             print(ii, jj)
                #             pdb.set_trace()
                #
                # import torch.autograd as autograd
                # # dia_grad = autograd.grad(batch_prob[0, 0, 2], batch_feature_dia, allow_unused=True)[0]
                # dia_grad = autograd.grad(batch_prob[2, 2, 155], batch_feature_dia, allow_unused=True)[0]
                # inter_dia = dia_grad[2] * batch_feature_dia[2]
                #
                # pro_grad = autograd.grad(batch_prob[2, 2, 155], batch_feature_pro, allow_unused=True)[0]
                # inter_pro = pro_grad[2] * batch_feature_pro[2]
                #
                # print(inter_dia, inter_pro)
                #
                # pdb.set_trace()

                batch_weight /= batch_weight.mean()
                loss_train = (batch_weight * torch.mean(criterion(batch_prob, batch_label), dim=-1)).mean()  # (N, T)
                loss_train.backward(retain_graph=True)
                loss_sum += loss_train.sum()
                train_size += to_npy(batch_len).sum()
                optimizer.step()
                optimizer.zero_grad()

                loss_cor = calculate_hsic2(batch_feature_dia, batch_feature_pro, batch_weight, device).mean()
                loss_cor.backward()
                cor_opt.step()
                cor_opt.zero_grad()
                corr.append(loss_cor.item())

                bar.update(batch_dia.shape[0])
            bar.close()

            print('training loss is: {}'.format(loss_sum / train_size))
            print('HSIC = {}'.format(mean(corr)))
            # print(calculate_hsic2(batch_feature_dia, batch_feature_pro, batch_weight, device))
            print(batch_weight[0:2])

            self.model.eval()

            '''
            # calculating training recall rates
            print('calculating training acc...')
            bar = tqdm(total=self.train_data.shape[0])
            for step, item in enumerate(self.train_loader):
                batch_input, batch_label, batch_len = item
                batch_prob = self.model(batch_input)  # (N, T, U)
                acc_train += calculate_acc(batch_prob, batch_label, batch_len)
                bar.update(batch_input.shape[0])
            bar.close()
            acc_train = np.array(acc_train) / train_size
            print('epoch:{}, train_acc:{}'.format(self.start_epoch + t, acc_train))
            '''

            # calculating validation recall rates
            print('calculating validation acc...')
            bar = tqdm(total=self.valid_dia.shape[0])
            for step, item in enumerate(self.valid_loader):
                batch_dia, batch_pro, batch_label, batch_len, _ = item
                batch_prob_dia, batch_feature_dia = self.model(batch_dia, 'dia')
                batch_prob_pro, batch_feature_pro = self.model(batch_pro, 'pro')
                batch_prob = batch_prob_dia + batch_prob_pro  # (N, T, U)
                acc_valid += calculate_acc(batch_prob, batch_label, batch_len)
                ndcg_valid += calculate_ndcg(batch_prob, batch_label, batch_len)
                valid_size += to_npy(batch_len).sum()
                bar.update(batch_dia.shape[0])
            bar.close()
            acc_valid = acc_valid / valid_size
            ndcg_valid = ndcg_valid / valid_size
            print('epoch:{}, valid_acc:{}, valid_ndcg:{}'.format(self.start_epoch + t, acc_valid, ndcg_valid))

            # calculating testing recall rates
            print('calculating testing acc...')
            bar = tqdm(total=self.test_dia.shape[0])
            for step, item in enumerate(self.test_loader):
                batch_dia, batch_pro, batch_label, batch_len, _ = item
                batch_prob_dia, batch_feature_dia = self.model(batch_dia, 'dia')
                batch_prob_pro, batch_feature_pro = self.model(batch_pro, 'pro')
                batch_prob = batch_prob_dia + batch_prob_pro  # (N, T, U)
                acc_test += calculate_acc(batch_prob, batch_label, batch_len)
                ndcg_test += calculate_ndcg(batch_prob, batch_label, batch_len)
                test_size += to_npy(batch_len).sum()
                bar.update(batch_dia.shape[0])
            bar.close()
            acc_test = np.array(acc_test) / test_size
            ndcg_test = np.array(ndcg_test) / test_size
            # group_test = hit / num
            print('epoch:{}, test_acc:{}, test_ndcg:{}'.format(self.start_epoch + t, acc_test, ndcg_test))

            self.records['ndcg_valid'].append(ndcg_valid)
            self.records['ndcg_test'].append(ndcg_test)
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(ndcg_valid):
                epoch = 0
                self.threshold = np.mean(ndcg_valid)
                best_acc, best_ndcg = acc_test, ndcg_test
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           'checkpoints/best_' + args.model + '_stable_' + args.data + '.pth')
            else:
                epoch += 1

            # if acc_valid does not increase, early stop it
            if self.early_max <= epoch:
                break

        print('Stop training!')
        print('Final test_acc:{}, test_ndcg:{}'.format(best_acc, best_ndcg))


if __name__ == '__main__':
    # load data and model
    insurance1 = 'Medicare'
    insurance2 = 'Other' if args.data == 'mimic4' else 'Private'

    if args.data == 'mimic3' or args.data == 'cms':
        file = open('./data/' + args.data + '/dataset_' + insurance1 + '.pkl', 'rb')
        file_data = joblib.load(file)
        [data_dia, data_pro, label_seq, real_len] = file_data  # tensor (NUM, T, U)
    elif args.data == 'mimic4':
        data_dia = torch.tensor(np.load('./data/mimic4/data_dia_' + insurance1 + '.npy'))
        data_pro = torch.tensor(np.load('./data/mimic4/data_pro_' + insurance1 + '.npy'))
        label_seq = torch.tensor(np.load('./data/mimic4/label_dia_' + insurance1 + '.npy'))
        real_len = np.load('./data/mimic4/true_len_' + insurance1 + '.npy')
    else:
        raise NotImplementedError

    dia = data_dia.to(torch.float32)
    pro = data_pro.to(torch.float32)
    label = label_seq.to(torch.float32)
    length = torch.tensor(real_len)

    ix = np.linspace(0, len(dia) - 1, len(dia))
    np.random.seed(0)  # fix data divide seed if needed
    np.random.shuffle(ix)
    train_ratio, dev_ratio = (0.7, 0.3)
    end = int(train_ratio * len(dia))

    train_dia = dia[ix[:end]]
    valid_dia = dia[ix[end:]]
    train_pro = pro[ix[:end]]
    valid_pro = pro[ix[end:]]
    train_label = label[ix[:end]]
    valid_label = label[ix[end:]]
    train_length = length[ix[:end]]
    valid_length = length[ix[end:]]

    if args.data == 'mimic3' or args.data == 'cms':
        file = open('./data/' + args.data + '/dataset_' + insurance2 + '.pkl', 'rb')
        file_data = joblib.load(file)
        [data_dia, data_pro, label_seq, real_len] = file_data  # tensor (NUM, T, U)
    elif args.data == 'mimic4':
        data_dia = torch.tensor(np.load('./data/mimic4/data_dia_' + insurance2 + '.npy'))
        data_pro = torch.tensor(np.load('./data/mimic4/data_pro_' + insurance2 + '.npy'))
        label_seq = torch.tensor(np.load('./data/mimic4/label_dia_' + insurance2 + '.npy'))
        real_len = np.load('./data/mimic4/true_len_' + insurance2 + '.npy')
    else:
        raise NotImplementedError

    test_dia = dia.to(torch.float32)
    test_pro = pro.to(torch.float32)
    test_label = label.to(torch.float32)
    test_length = torch.tensor(length)

    data_combo = [train_dia, valid_dia, test_dia, train_pro, valid_pro, test_pro,
                  train_label, valid_label, test_label, train_length, valid_length, test_length]

    if args.model == 'lstm':
        import Stable.models.lstm
        model = Stable.models.lstm.LSTM(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                        hidden_size=args.hidden_size,
                                        dropout=args.drop, batch_first=True)
    elif args.model == 'retain':
        import Stable.models.retain
        model = Stable.models.retain.RETAIN(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                            hidden_size=args.hidden_size,
                                            dropout=args.drop, batch_first=True)

    elif args.model == 'dipole':
        import Stable.models.dipole
        model = Stable.models.dipole.Dipole(attention_type='location_based', icd_size=data_dia.shape[2],
                                            pro_size=data_pro.shape[2], attention_dim=16,
                                            hidden_size=args.hidden_size, dropout=args.drop,
                                            batch_first=True)
    elif args.model == 'stagenet':
        args.hidden_size = 384
        import Stable.models.stagenet
        model = Stable.models.stagenet.StageNet(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                                hidden_size=args.hidden_size, dropout=args.drop)

    elif args.model == 'setor':
        import Stable.models.setor
        model = Stable.models.setor.SETOR(alpha=0.5, hidden_size=args.hidden_size,
                                          intermediate_size=args.hidden_size, hidden_act='relu',
                                          icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                          max_position=10000, num_attention_heads=2, num_layers=1, dropout=args.drop)

    elif args.model == 'concare':
        import Stable.models.concare
        model = Stable.models.concare.ConCare(icd_size=data_dia.shape[2], pro_size=data_pro.shape[2],
                                              hidden_dim=args.hidden_size, MHD_num_head=4, drop=args.drop)

    else:
        raise NotImplementedError

    num_params = 0

    # for name in model.state_dict():
    #     print(name)

    for param in model.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    load = False

    if load:
        checkpoint = torch.load('checkpoints/best_' + args.model + '_stable_' + args.data + '.pth')
        model.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'ndcg_valid': [], 'ndcg_test': []}
        start = time.time()

    trainer = Trainer(model, data_combo, records)
    trainer.train()
