<div align="center">
  
# Modelling Alzheimer's Disease progression<br/>in longitudinal fMRI with a Contrastive VAE

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
[![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?style=for-the-badge&logo=numpy&logoColor=fff)](#)
![nilearn](https://img.shields.io/badge/nilearn-53696A?style=for-the-badge)

</div>

> Disentangling Alzheimer's-specific neurodegeneration from healthy ageing 
> using a Contrastive Variational Autoencoder trained on longitudinal fMRI 
> functional connectivity data (ADNI-2, ADNI-3).

Supervised by Dr. Yu Zhang, Brain Imaging and Computation Lab, Lehigh University

## Code overview

- Encodes resting-state FC matrices into two latent subspaces:
  **z** (shared variance) and **s** (AD-specific variance)
- Incorporates age as an auxiliary regression objective to separate 
  chronological from biological brain age
- Encourages disentanglement via total correlation (TC) loss + discriminator
- Trained on **1,612 subjects** across three diagnostic groups: CN, MCI, De

## Loss function

| Component | Purpose |
|---|---|
| Reconstruction | FC matrix fidelity |
| KL divergence | Latent space regularisation |
| TC loss | Disentanglement between z and s |
| Discriminator | Enforce independence of latent subspaces |
| **Age (MSE)** | **Supervise biological age in each branch** |

