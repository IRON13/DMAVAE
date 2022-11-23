import logging
import pandas as pd
import numpy as np
import torch
import pyro
from dmavae_gpu import DMA_VAE
from load_datasets import load_data

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)

def run(N, num):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    data = load_data(path="./Data", N = N, num = num)
    (x_data, m_data, t_data, y_data) = data

    x_data = x_data.float()
    m_data = m_data.float()
    t_data = t_data.float()
    y_data = y_data.float()

    # Train.
    pyro.clear_param_store()
    dmavae = DMA_VAE(feature_dim=3,
                    latent_Ztm_dim=1,
                    latent_Zty_dim=1,
                    latent_Zmy_dim=1,
                    hidden_dim=128,
                    num_layers=4,
                    num_samples=10)
    dmavae.fit(x_data, m_data, t_data, y_data,
               num_epochs=50,
               batch_size=128,
               learning_rate=1e-3,
               learning_rate_decay=0.01, weight_decay=1e-4)

    NDE, NIEr, NIE, ATE = dmavae.effect_estimation(x_data)

    NDE = NDE.cpu().detach().numpy()
    NIEr = NIEr.cpu().detach().numpy()
    NIE = NIE.cpu().detach().numpy()
    ATE = ATE.cpu().detach().numpy()

    NDE_res = np.mean(NDE)
    NIEr_res = np.mean(NIEr)
    NIE_res = np.mean(NIE)
    ATE_res = np.mean(ATE)

    return NDE_res, NIEr_res, NIE_res, ATE_res

N_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
res_NDE = []
res_NIEr = []
res_NIE = []
res_ATE = []

for N in N_list:
    for i in range(30):
        print(N)
        NDE, NIEr, NIE, ATE = run(N, i)
        res_NDE.append(NDE)
        res_NIEr.append(NIEr)
        res_ATE.append(ATE)
        print("Finished:", str(i + 1))

    res = pd.DataFrame({'res_ATE': res_ATE, 'res_NDE': res_NDE, 'res_NIEr': res_NIEr})
    res.to_csv(r"./Res_Performance/Syn_DMAVAE_" + str(N) + ".csv",sep=',',float_format='%.3f')

    res_NDE = []
    res_NIEr = []
    res_NIE = []
    res_ATE = []







