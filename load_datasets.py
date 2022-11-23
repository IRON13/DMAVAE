import torch
import numpy as np

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def load_data(path, N, num):
    path_data = path

    data = np.loadtxt(path_data + '/Syn_' + str(N) + '/Syn_' + str(N) + '_' + str(num+1) + '.csv', delimiter=',', skiprows=1)
    data[:, 1:4] = standardization(data[:, 1:4])

    t, y = data[:, 0], data[:, 5][:, np.newaxis]
    m = data[:, 4]
    x = data[:, 1:4]

    x = torch.from_numpy(x)
    m = torch.from_numpy(m).squeeze()
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()

    # x = x.cuda()
    # m = m.cuda()
    # y = y.cuda()
    # t = t.cuda()

    data = (x, m, t, y)

    return data


