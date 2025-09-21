import random
import pandas as pd
import numpy as np


from numpy import array
from numpy import argmax
from torch.utils.data import Dataset



class ViewDataMO(Dataset):

    def __init__(self, df_X, df_y):
        super(ViewDataMO, self).__init__()
        # df_X = pd.read_csv(df_X)
        tasks = df_y.columns[1:] #The number of class_embed_vectors is the same as that of Tasks, and the order is also the same
        self.X = df_X.iloc[:, 2:].to_numpy().astype(np.float32) # Remove the data after the columns 'Drug', 'smiles', and 'label'
        # self.smiles = df_y['smiles'].tolist()
        self.smiles = df_y['smiles'].tolist()
        self.data = []

        # Traverse the data under each ADR category and form a triplet of (com_X, adr_X, label)
        for index, row in df_y.iterrows():
            for i,task in enumerate(tasks):
                if pd.notna(row[task]):
                    smiles = row['smiles']
                    smiles_embed = df_X[df_X['smiles'] == smiles].iloc[:, 2:].to_numpy().astype(np.float32).reshape(-1)
                    # class_embed = class_embed_vector[i]
                    label = row[task]
                    # self.data.append((smiles, smiles_embed, class_embed, label, i, task))
                    self.data.append((smiles, smiles_embed, label, i, task)) # "task" is the name of the ADR category

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)
    

# ViewDataMO_gptemb
class ViewDataMO_gptemb(Dataset):

    def __init__(self, df_X, df_y):
        super(ViewDataMO_gptemb, self).__init__()
        # df_X = pd.read_csv(df_X)
        tasks = df_y.columns[1:]
        self.X = df_X.iloc[:, 2:].to_numpy().astype(np.float32)
        self.smiles = df_y['smiles'].tolist()
        self.data = []

        for index, row in df_y.iterrows():
            for i,task in enumerate(tasks):
                if pd.notna(row[task]):
                    smiles = row['smiles']
                    smiles_embed = df_X[df_X['smiles'] == smiles].iloc[:, 2:].to_numpy().astype(np.float32).reshape(-1)
                    label = row[task]
                    self.data.append((smiles, smiles_embed, label, i, task))

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)


def separate_active_and_inactive_data (datafarme,  tasks):
    dataset_pos = []
    dataset_neg = []

    for task in tasks:
        a = datafarme[['smiles' , task]]
        b = a.loc[a[task]==0]
        a = a.loc[a[task]==1]
        dataset_pos.append(a)
        dataset_neg.append(b)

    return dataset_pos, dataset_neg


def count_lablel(dataset):

    label_pos = 0
    lable_neg = 0

    data_pos = []
    data_neg = []

    for i, data in enumerate(dataset):
        smiles, embbed_drug, lbl, task_number, task_name= data
        if lbl == 1.:
            label_pos += 1
            data_pos.append(data)
        else:
            lable_neg += 1
            data_neg.append(data)

    return label_pos, lable_neg , data_pos, data_neg



def compute_metrics(y_true, y_pred):
    from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return f1, mcc, bacc