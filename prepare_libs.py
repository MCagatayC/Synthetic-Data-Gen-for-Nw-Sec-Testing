import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Ortak ayarlar
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Dataset yolunu belirt
DATASET_PATH = '/home/dolly/.cache/kagglehub/datasets/ravikumargattu/network-traffic-dataset/versions/2'
CSV_NAME = os.path.join(DATASET_PATH, 'Midterm_53_group.csv') 

# Veriyi yükle ve ölçekle (isteğe bağlı)
def load_and_scale_data(numeric_only=True):
    df = pd.read_csv(os.path.join(DATASET_PATH, CSV_NAME))
    if numeric_only:
        df = df.select_dtypes(include=[np.number])
    df = df.dropna().reset_index(drop=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return df, scaled_data, scaler
