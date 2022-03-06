import pdb
import math
import torch
import warnings

warnings.filterwarnings('ignore')

GPU = 1
device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
# print('function_device:{}'.format(device))
# print('function_gpu:{}'.format(GPU))


def calculate_hsic(dia_feat, pro_feat, weight, gpu):
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

    pdb.set_trace()

    return hsic.reshape(batch_size, time_size)


def random_fourier_mapping(x, n=5):
    x = x.unsqueeze(-1)
    while len(x.shape) < 3:
        x = x.unsqueeze(0)

    batch_time, feature, _ = x.shape
    omega = torch.normal(mean=0, std=1, size=(batch_time, feature, n)).to(device)
    phi = torch.rand(size=(batch_time, feature, n)).to(device) * 2 * math.pi
    return math.sqrt(2)*torch.cos(omega*x+phi)


def linear_covariance(dia_feat, pro_feat):
    # dia_feat & pro_feat [N, T, 2H] --> [N*T, 2H]
    batch_size, time_size, feat_size = dia_feat.shape
    dia_feat = dia_feat.view(batch_size * time_size, -1)
    pro_feat = pro_feat.view(batch_size * time_size, -1)

    # the size of sum is [2H]
    dia_fourier_sum = torch.sum(dia_feat, dim=0) / (batch_size * time_size)
    pro_fourier_sum = torch.sum(pro_feat, dim=0) / (batch_size * time_size)
    reg = 0

    matrix = 0  # [2H, 2H]
    for j in range(batch_size*time_size):
        dia_vec = (dia_feat[j] - dia_fourier_sum).unsqueeze(-1)  # [2H, 1]
        pro_vec = (pro_feat[j] - pro_fourier_sum).unsqueeze(-1)
        matrix += torch.matmul(dia_vec.T, pro_vec) / (batch_size * time_size - 1)
    print(matrix)
    pdb.set_trace()

    return dia_feat, pro_feat


def cross_covariance(dia_feat, pro_feat, n=5):
    # dia_feat & pro_feat [N, T, 2H] --> [N*T, 2H]
    batch_size, time_size, feat_size = dia_feat.shape
    dia_feat = dia_feat.view(batch_size * time_size, -1)
    pro_feat = pro_feat.view(batch_size * time_size, -1)

    # the size of sum is [2H, n]
    dia_fourier_sum = torch.sum(random_fourier_mapping(dia_feat, n), dim=0) / (batch_size * time_size)
    pro_fourier_sum = torch.sum(random_fourier_mapping(pro_feat, n), dim=0) / (batch_size * time_size)
    reg = 0

    for i in range(feat_size):
        matrix = torch.zeros(n, n).to(device)  # [n. n]
        # need to add codes here
            
        for k in range(batch_size*time_size):
            dia_vec = (random_fourier_mapping(dia_feat[k, i], n) - dia_fourier_sum[i, :])[0, :]
            pro_vec = (random_fourier_mapping(pro_feat[k, i], n) - pro_fourier_sum[i, :])[0, :]
            matrix += torch.matmul(dia_vec.T, pro_vec) / (batch_size * time_size - 1)
        print(torch.norm(matrix, p='fro'))

    return dia_feat, pro_feat


if __name__ == '__main__':
    xx = torch.randn(32, 10, 4, ).to(device)
    yy = xx + 0.01 * torch.randn(32, 10, 4, ).to(device)
    # linear_covariance(torch.rand(32, 10, 4), torch.rand(32, 10, 4))  # --> 0.05263
    # linear_covariance(torch.ones(32, 10, 4), torch.ones(32, 10, 4))  # --> 0
    cross_covariance(torch.randn(32, 10, 4, ).to(device), torch.randn(32, 10, 4, ).to(device))
    cross_covariance(xx, yy)
    # h_value = calculate_hsic(torch.randn(32, 10, 4,).to(device), torch.randn(32, 10, 4,).to(device))  # --> 0.0965
    # h_value = calculate_hsic(xx, yy)  # --> 0.4227
    pdb.set_trace()
