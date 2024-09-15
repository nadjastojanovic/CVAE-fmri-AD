# %% imports
import numpy as np

import scipy
import scipy.io
from scipy.stats import pearsonr

import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from nilearn.connectome import ConnectivityMeasure

# %% helper functions for data pre-processing

def get_fc(ts, metric = 'correlation'):
    if metric == 'correlation':
        fc2 = np.corrcoef(ts.T)
        np.fill_diagonal(fc2, 0)
        transf_fc = np.arctanh(fc2)
    else: 
        measure = ConnectivityMeasure(kind='partial correlation') 
        connectivities = measure.fit_transform([ts])[0]
        np.fill_diagonal(connectivities, 0)
        transf_fc = np.arctanh(connectivities)
    return transf_fc

def keep_triangle_half(num_edge, num_sub_total, all_mats1, connect_type = 'roi'):
    all_edges = np.zeros((num_sub_total, num_edge))
    if connect_type == 'roi':
        idx = np.triu_indices_from(all_mats1[0, :, :], 1)  
    elif connect_type == 'net':
        idx = np.triu_indices_from(all_mats1[0, :, :], 0)
    for i_sub in range(num_sub_total):
        all_edges[i_sub, :] = all_mats1[i_sub,:, :][idx]
    return all_edges

def process_fc_data(ts_data, offset=0, dataset=2):
    idx = 0

    for i in range(len(ts_data)):
        if (dataset == 2 and ts_data[i][0][62].size > 0) or (dataset == 3 and ~np.isnan(adni3['age'])[0][i]):
            fc_matrix = get_fc(ts_data[i][0][0]) if dataset == 2 else get_fc(ts_data[i][:, :100])
            label = ts_data[i][0][62][0][:2] if dataset == 2 else adni3['dx'][i][:2]
                
            for key in indices.keys():
                if label == key:
                    indices[key].append(idx + offset)
                    fc_matrices.append(fc_matrix)
                    idx += 1
                    break
                
def assign_ages(indices, age_array, threshold):
    arr = np.zeros(len(indices))
    
    for idx, x in enumerate(indices):
        if idx < threshold:
            arr[idx] = age_array[0][x][0][10][0][0]
        else:
            arr[idx] = age_array[1][0][x]
    
    return arr

# %% data pre-processing

adni2 = scipy.io.loadmat('/Users/nadastojanovic/Development/BIC/adni2.mat')
adni3 = scipy.io.loadmat('/Users/nadastojanovic/Development/BIC/adni3_more.mat')

fc_matrices = []
indices = {'CN': [], 'MC': [], 'De': []}

process_fc_data(adni2['ans'])
process_fc_data(adni3['fc'], offset=1153, dataset=3)

fc_matrices = np.array(fc_matrices)
fc_triangles = keep_triangle_half(4950, 1612, fc_matrices)

del indices['CN'][118] # one pesky NA value

ADNI_cn = fc_triangles[indices['CN']]
ADNI_mc = fc_triangles[indices['MC']] 
ADNI_ad = fc_triangles[indices['De']] 

ADNI_cn_age = assign_ages(indices['CN'], [adni2['ans'], adni3['age']], 528)
ADNI_mc_age = assign_ages(indices['MC'], [adni2['ans'], adni3['age']], 455)
ADNI_ad_age = assign_ages(indices['De'], [adni2['ans'], adni3['age']], 169)

# %% model architecture

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, bias=True):
        super(Encoder, self).__init__()
        filters = 2000
        intermediate_dim = 500

        self.z_lay1 = nn.Linear(input_shape[1], filters)
        self.z_h_layer = nn.Linear(filters, intermediate_dim)
        
        self.z_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(intermediate_dim, latent_dim)

        self.s_lay1 = nn.Linear(input_shape[1], filters)
        self.s_h_layer = nn.Linear(filters, intermediate_dim)
        
        self.s_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.s_log_var_layer = nn.Linear(intermediate_dim, latent_dim)
        
        self.age_layer_z = nn.Linear(intermediate_dim, 1)
        self.age_layer_s = nn.Linear(intermediate_dim, 1)
        
    def forward(self, x):
        z_h = F.relu(self.z_lay1(x))
        z_h = F.relu(self.z_h_layer(z_h))
        z_mean = self.z_mean_layer(z_h)
        z_log_var = self.z_log_var_layer(z_h)

        s_h = F.relu(self.s_lay1(x))
        s_h = F.relu(self.s_h_layer(s_h))
        s_mean = self.s_mean_layer(s_h)
        s_log_var = self.s_log_var_layer(s_h)
        
        z_age_output = self.age_layer_z(z_h)
        s_age_output = self.age_layer_s(s_h)

        return z_mean, z_log_var, s_mean, s_log_var, z_age_output, s_age_output
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape, bias=True):
        super(Decoder, self).__init__()
        intermediate_dim = 500
        self.output_shape = output_shape

        self.decoder_input = nn.Linear(latent_dim, intermediate_dim)
        self.decoder_output = nn.Linear(intermediate_dim, output_shape[1])
        
    def forward(self, z):
        x = F.relu(self.decoder_input(z))
        x = F.leaky_relu(self.decoder_output(x))
        # x = F.hardtanh(self.decoder_output(x),-2,2)
        return x
  
class CVAE(nn.Module):
    def __init__(self, input_shape, latent_dim, disentangle, bias=True):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_shape, latent_dim, bias)
        self.decoder = Decoder(latent_dim * 2, input_shape, bias)
        
        self.disentangle = disentangle

        if self.disentangle:
            self.discriminator = nn.Linear(latent_dim * 2, 1)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, tg_inputs, bg_inputs):
        tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, _, _ = self.encoder(tg_inputs)
        bg_z_mean, bg_z_log_var, _, _, _, _ = self.encoder(bg_inputs)

        tg_z = self.reparameterize(tg_z_mean, tg_z_log_var)
        tg_s = self.reparameterize(tg_s_mean, tg_s_log_var)
        bg_z = self.reparameterize(bg_z_mean, bg_z_log_var)

        tg_outputs = self.decoder(torch.cat([tg_z, tg_s], dim=-1))
        bg_outputs = self.decoder(torch.cat([torch.zeros_like(tg_z), bg_z], dim=-1))

        return tg_outputs, bg_outputs
  
input_shape = (1612, 4950)
latent_dim = 50
batch_size = 64
disentangle = True

cvae = CVAE(input_shape, latent_dim, disentangle)

optimizer = torch.optim.Adam(cvae.parameters(), lr = 0.001)

# %% model training helper functions
def loss_function(tg_s_age_output, bg_z_age_output, tg_inputs_batch, bg_inputs_batch, tg_outputs, bg_outputs, tg_z_log_var, tg_s_log_var, bg_z_log_var, tg_z_mean, tg_s_mean, bg_z_mean):
    total_corr = 0.0
    
    # R squared
    r2_s = r2_score(tg_s_age_output.detach().numpy(), tg_inputs_batch[:, -1])
    r2_z = r2_score(bg_z_age_output.detach().numpy(), bg_inputs_batch[:, -1])
    
    # Pearson correlation between raw and reconstructed FC
    for idx in range(len(tg_inputs_batch)):
        raw_fc = tg_inputs_batch[idx, :-1].numpy()
        reconstructed_fc = tg_outputs[idx].detach().numpy()
        corr, _ = pearsonr(raw_fc, reconstructed_fc)
        total_corr += corr/len(tg_inputs_batch)
    
    # mse loss for age
    age_loss = F.mse_loss(tg_s_age_output, tg_inputs_batch[:, -1].float().unsqueeze(1))
    age_loss += F.mse_loss(bg_z_age_output, bg_inputs_batch[:, -1].float().unsqueeze(1))
    
    # mse loss
    reconstruction_loss = F.mse_loss(tg_outputs, tg_inputs_batch[:, :-1].float())
    reconstruction_loss += F.mse_loss(bg_outputs, bg_inputs_batch[:, :-1].float())
    
    # KL divergence loss
    kl_loss = 1 + tg_z_log_var - tg_z_mean.pow(2) - tg_z_log_var.exp()
    kl_loss += 1 + tg_s_log_var - tg_s_mean.pow(2) - tg_s_log_var.exp()
    kl_loss += 1 + bg_z_log_var - bg_z_mean.pow(2) - bg_z_log_var.exp()
    kl_loss = torch.sum(kl_loss, dim=-1)
    kl_loss *= -0.5
    
    # tc losst
    z1 = tg_z_mean[:tg_inputs_batch.size(0) // 2, :]
    z2 = tg_z_mean[tg_inputs_batch.size(0) // 2:, :]
    s1 = tg_s_mean[:tg_inputs_batch.size(0) // 2, :]
    s2 = tg_s_mean[tg_inputs_batch.size(0) // 2:, :]
    
    if cvae.disentangle and s1.size(0) == s2.size(0):
        q_bar = torch.cat([torch.cat([s1, z2], dim=1), torch.cat([s2, z1], dim=1)], dim=0)
        q = torch.cat([torch.cat([s1, z1], dim=1), torch.cat([s2, z2], dim=1)], dim=0)
        
        q_bar_score = (F.sigmoid(cvae.discriminator(q_bar)) + 0.1) * 0.85
        q_score = (F.sigmoid(cvae.discriminator(q)) + 0.1) * 0.85
        
        discriminator_loss = - torch.log(torch.abs(q_score)) - torch.log(torch.abs(1 - q_bar_score))
        
        tc_loss = torch.log(q_score / (1 - q_score))
    else:
        discriminator_loss = torch.tensor(0)
        tc_loss = torch.tensor(0)

    return reconstruction_loss, kl_loss, tc_loss, discriminator_loss, age_loss, r2_s, r2_z, total_corr

def write(total_loss, total_corr, r2_s_total, r2_z_total, length):
    print(f'Loss: {total_loss / length:.4f} \
    \nCorrelation: {total_corr / length:.4f} \
    \nR2 tg: {r2_s_total / length:.2f} R2 bg: {r2_z_total / length:.2f}')

# %% model training setup

# age cn into z, mci into s
ADNI_cn = np.concatenate([ADNI_cn, ADNI_cn_age.reshape(-1, 1)], axis=1)
ADNI_mc = np.concatenate([ADNI_mc, ADNI_mc_age.reshape(-1, 1)], axis=1)

train_data_cn, test_data_cn = train_test_split(ADNI_cn[:631], test_size=0.2)
train_data_mc, test_data_mc = train_test_split(ADNI_mc, test_size=0.2)

train_dataset = TensorDataset(torch.tensor(train_data_cn), torch.tensor(train_data_mc))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(torch.tensor(test_data_cn), torch.tensor(test_data_mc))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
epochs = 50

a = 1
b = 1
g = 1
d = 1
e = 1

train_losses = np.zeros((epochs, 7))
test_losses = np.zeros((epochs, 7))

# %% model training

for epoch in range(epochs):
    cvae.train()
    
    print(f'Epoch: {epoch+1}/{epochs}\n')
    
    train_total_loss = 0.0
    train_r2_s_total = 0.0
    train_r2_z_total = 0.0

    # TRAIN
    for batch_idx, (bg_inputs_batch, tg_inputs_batch) in enumerate(train_loader):
        optimizer.zero_grad()

        tg_outputs, bg_outputs = cvae(tg_inputs_batch[:, :-1].float(), bg_inputs_batch[:, :-1].float())
        tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, _, tg_s_age_output = cvae.encoder(tg_inputs_batch[:, :-1].float())
        bg_z_mean, bg_z_log_var, _, _, bg_z_age_output, _ = cvae.encoder(bg_inputs_batch[:, :-1].float())
        
        reconstruction_loss, kl_loss, tc_loss, discriminator_loss, age_loss, r2_s, r2_z, train_total_corr = loss_function(tg_s_age_output, bg_z_age_output, tg_inputs_batch, bg_inputs_batch, tg_outputs, bg_outputs, tg_z_log_var, tg_s_log_var, bg_z_log_var, tg_z_mean, tg_s_mean, bg_z_mean)
        
        train_losses[epoch][0] += reconstruction_loss
        train_losses[epoch][1] += torch.mean(kl_loss)
        train_losses[epoch][2] += torch.mean(tc_loss.float())
        train_losses[epoch][3] += torch.mean(discriminator_loss.float())
        train_losses[epoch][4] += age_loss
        
        train_r2_s_total += r2_s
        train_r2_z_total += r2_z
        
        loss = a * reconstruction_loss + b * kl_loss + g * tc_loss + d * discriminator_loss + e * age_loss
        loss = torch.mean(loss)
        
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        
    train_losses[epoch][5] = train_r2_s_total
    train_losses[epoch][6] = train_r2_z_total
    train_losses[epoch] /= len(train_loader)
        
    write(train_total_loss, train_total_corr, train_r2_s_total, train_r2_z_total, len(train_loader))
    print('   -------------------   ')
        
    test_r2_s_total = 0.0
    test_r2_z_total = 0.0
    test_total_loss = 0.0
            
    # TEST
    for batch_idx, (bg_inputs_batch, tg_inputs_batch) in enumerate(test_loader):
        tg_outputs, bg_outputs = cvae(tg_inputs_batch[:, :-1].float(), bg_inputs_batch[:, :-1].float())
        tg_z_mean, tg_z_log_var, tg_s_mean, tg_s_log_var, _, tg_s_age_output = cvae.encoder(tg_inputs_batch[:, :-1].float())
        bg_z_mean, bg_z_log_var, _, _, bg_z_age_output, _ = cvae.encoder(bg_inputs_batch[:, :-1].float())
        
        reconstruction_loss, kl_loss, tc_loss, discriminator_loss, age_loss, r2_s, r2_z, test_total_corr = loss_function(tg_s_age_output, bg_z_age_output, tg_inputs_batch, bg_inputs_batch, tg_outputs, bg_outputs, tg_z_log_var, tg_s_log_var, bg_z_log_var, tg_z_mean, tg_s_mean, bg_z_mean)
        
        test_losses[epoch][0] += reconstruction_loss
        test_losses[epoch][1] += torch.mean(kl_loss)
        test_losses[epoch][2] += torch.mean(tc_loss.float())
        test_losses[epoch][3] += torch.mean(discriminator_loss.float())
        test_losses[epoch][4] += age_loss
        
        test_r2_s_total += r2_s
        test_r2_z_total += r2_z
        
        loss = a * reconstruction_loss + b * kl_loss + g * tc_loss + d * discriminator_loss + e * age_loss
        loss = torch.mean(loss)

        test_total_loss += loss.item()
        
    test_losses[epoch][5] = test_r2_s_total
    test_losses[epoch][6] = test_r2_z_total
    test_losses[epoch] /= len(test_loader)
    
    write(test_total_loss, test_total_corr, test_r2_s_total, test_r2_z_total, len(test_loader))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    
titles = ['Reconstruction Loss', 'KL Loss', 'TC Loss', 'Discriminator Loss', 'Age Loss', 'R2 Target', 'R2 Background']
for i in range(7):
    fig, ax = plt.subplots()
    ax.plot(range(2, epochs), train_losses[2:, i], label='Train Loss', color='blue')
    ax.plot(range(2, epochs), test_losses[2:, i], label='Test Loss', color='orange')
    ax.set_title(f'{titles[i]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()

# done:
#
# * use simple model like SVM to predict age in cross-validation
#   (5 fold or 10 fold) to see maximum R2 model performance,
#   and decide if task is too difficult or if I just need to tune parameters
# ^ conclusion: R2 > 0.6, task is not too difficult, need to tune parameters
#
# * simplified data pre-processing via helper functions
# ^ benefit: just so much better to look at, gave me satisfaction
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# to do:
# combine dementia and mci after code runs
# expand using ADNI 3 data
# cross validation framework
# parameter tunning, might need to add weights to the components of mse losses (goal: maximize R2 on test set)


# ridge & lasso regression to minimalize overfitting

# incorporate more data ->
# cross validation framework ->
# tune learning rate, dropout, L1 & L2 regularization
    
# %% simple age regression model via SVM

X = ADNI_mc[:, :-1] # fc
y = ADNI_mc[:, -1]  # age

kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse = []
r2 = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svm_model = SVR(kernel='linear') 
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    
    mse.append(mean_squared_error(y_test, y_pred))
    r2.append(r2_score(y_test, y_pred))
    
print("Average MSE: ", np.mean(mse))    # 0.026429938273756233
print("Average R2: ", np.mean(r2))      # 0.6204682991263251
