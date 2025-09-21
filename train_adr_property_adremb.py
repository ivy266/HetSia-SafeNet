import torch
import pandas as pd
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
import argparse
import os

from numpy import array
from numpy import argmax
from tensorflow.keras.callbacks import  History

from utils.general import ViewDataMO, compute_metrics, separate_active_and_inactive_data
from utils.general import count_lablel

from model.heterogeneous_siamese_sider import siamese_model_sider_adr_emb
import pathlib


from sklearn.model_selection import KFold
from collections import defaultdict
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


np.random.seed(2025) #2025
tf.random.set_seed(2025)

def evaluate_model(args, df, k, shuffle, l_test, task_index_test, lbls_test):
    
    result_valid = defaultdict(list)
    result_test = defaultdict(list)
    s = 0
    
    args.view_list == "AC50,Cmax"
    dim_drug_x = 389
    # if args.view_list == "AC50":
    #     dim_drug_x = 194
    # elif args.view_list == "Cmax":
    #     dim_drug_x = 195
    # elif args.view_list == "AC50,Cmax":
    #     dim_drug_x = 389
    # elif args.view_list == "AC50,Cmax,disrupt":
    #     dim_drug_x = 389
    # elif args.view_list == "AC50,Cmax,ECFP4":
    #     dim_drug_x = 1413
    # elif args.view_list == "ECFP4":
    #     dim_drug_x = 1024
    
    kf = KFold(n_splits=k, shuffle=shuffle, random_state=42)
    k = 0
    for train_index, test_index in kf.split(df):
        print(f'Now is Fold: {k+1}')
        k += 1
        train_ds = [df[index] for index in train_index] 
        
        valid_ds = [df[index] for index in test_index]
        
        label_pos, label_neg, _, _ = count_lablel(train_ds)
        print(f'train positive label: {label_pos} - train negative label: {label_neg}')

        label_pos, label_neg, _, _ = count_lablel(valid_ds)
        print(f'Test positive label: {label_pos} - Test negative label: {label_neg}')

        l_train = []
        # r_train = []
        lbls_train = []
        task_index_train = []
        l_valid = []
        # r_valid = []
        lbls_valid = []
        task_index_valid = []

        for i , data in enumerate(train_ds):
            smiles, embbed_drug, lbl, task_number, task_name = data
            # embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_train.append(embbed_drug)
            # r_train.append(embbed_task)
            lbls_train.append(lbl)
            task_index_train.append(task_number)
        
        for i , data in enumerate(valid_ds):
            smiles, embbed_drug, lbl, task_number, task_name = data
            # embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_valid.append(embbed_drug)
            # r_valid.append(embbed_task)
            lbls_valid.append(lbl)
            task_index_valid.append(task_number)

        l_train = np.array(l_train).reshape(-1,dim_drug_x,1)
        # r_train = np.array(r_train).reshape(-1,512,1)
        lbls_train = np.array(lbls_train)
        task_index_train = np.array(task_index_train).reshape(-1,1)

        l_valid = np.array(l_valid).reshape(-1,dim_drug_x,1)
        # r_valid = np.array(r_valid).reshape(-1,512,1)
        lbls_valid = np.array(lbls_valid)
        task_index_valid = np.array(task_index_valid).reshape(-1,1)

        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1) #25
        # model_checkpoint = ModelCheckpoint('best_model', monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
        model_checkpoint = ModelCheckpoint(f'/home/datahouse1/liujin/HetSia-SafeNet/result_split/best_model_emb_{k}', monitor='val_accuracy', save_best_only=True, verbose=1, save_format='tf')
        siamese_net = siamese_model_sider_adr_emb(dim_X=dim_drug_x,lr=args.lr)
        history = History()
        
        P = siamese_net.fit(
            [l_train, task_index_train], 
            lbls_train, 
            epochs=args.epoch_s, 
            batch_size=args.batch_size, #128
            validation_data=([l_valid, task_index_valid], lbls_valid), 
            callbacks=[early_stopping, model_checkpoint, history]
        ) # batch_size = 128 
        
       
        # Import the best model
        siamese_net = tf.keras.models.load_model(f'/home/datahouse1/liujin/HetSia-SafeNet/result_split/best_model_emb_{k}')
  
        score_v  = siamese_net.evaluate([l_valid,task_index_valid], lbls_valid, verbose=1) #metrics=["accuracy", "mae", "mse",tf.keras.metrics.AUC()]
        predict_v = siamese_net.predict([l_valid,task_index_valid], verbose=1)
        predict_v = np.array(predict_v).reshape(-1); predict_v_ = np.where(predict_v>0.5,1,0)
        lbls_valid = lbls_valid.reshape(-1)
        f1, mcc, bacc = compute_metrics(lbls_valid, predict_v_)
        a = score_v #metrics=["accuracy", "AUC","Precision","Recall"]
        print("Valid score is", a)
        result_valid["Acc"].append(score_v[1]); result_valid["AUC"].append(score_v[2]); result_valid["F1"].append(f1); result_valid["Mcc"].append(mcc);
        result_valid["Bacc"].append(bacc); result_valid["Precision"].append(score_v[3]); result_valid["Recall"].append(score_v[4])

        score_t = siamese_net.evaluate([l_test,task_index_test], lbls_test, verbose=1)
        predict_t = siamese_net.predict([l_test,task_index_test], verbose=1)
        predict_t = np.array(predict_t).reshape(-1); predict_t_ = np.where(predict_t>0.5,1,0)
        lbls_test = lbls_test.reshape(-1)
        f1_, mcc_, bacc_ = compute_metrics(lbls_test, predict_t_)
        b = score_t
        print("Test score is", b)
        result_test["Acc"].append(score_t[1]); result_test["AUC"].append(score_t[2]); result_test["F1"].append(f1_); result_test["Mcc"].append(mcc_);
        result_test["Bacc"].append(bacc_); result_test["Precision"].append(score_t[3]); result_test["Recall"].append(score_t[4])
        
        if score_t[2] > s : # Always keep the best model (based on the results of the test)
            best_model = siamese_net
            s = score_t[2]
            print("The best AUC is:", s)
            # print("Save_model")
            
        
    return result_valid, result_test, best_model


def model_train_and_eva(args, train_X, train_y, test_X, test_y, val_X, val_y, sider_tasks):
    df = pd.concat([train_y, test_y, val_y])
    df_positive, df_negative = separate_active_and_inactive_data(df, sider_tasks)

    for i,d in enumerate(zip(df_positive,df_negative)):
        print(f'{sider_tasks[i]}=> positive: {len(d[0])} - negative: {len(d[1])}')
    
    
    train_X_ = pd.concat([train_X, val_X])
    train_y_ = pd.concat([train_y, val_y])

    train_ds = ViewDataMO(train_X_, train_y_)
    test_ds = ViewDataMO(test_X, test_y)
    args.view_list == "AC50,Cmax"
    dim_drug_x = 389
    # if args.view_list == "AC50":
    #     dim_drug_x = 194
    # elif args.view_list == "Cmax":
    #     dim_drug_x = 195
    # elif args.view_list == "AC50,Cmax":
    #     dim_drug_x = 389
    # elif args.view_list == "AC50,Cmax,disrupt":
    #     dim_drug_x = 389
    # elif args.view_list == "AC50,Cmax,ECFP4":
    #     dim_drug_x = 1413
    # elif args.view_list == "ECFP4":
    #     dim_drug_x = 1024
    
    smiles_test = []
    task_name_list = []
    task_index_test = []
    l_test = []
    # r_test = []
    lbls_test = []

    for i , data in enumerate(test_ds):
        smiles, embbed_drug, lbl, task_number, task_name = data
        l_test.append(embbed_drug)
        # r_test.append(embbed_task)
        lbls_test.append(lbl)
        smiles_test.append(smiles)
        task_index_test.append(task_number)
        task_name_list.append(task_name)
    l_test = np.array(l_test).reshape(-1,dim_drug_x,1)
    # r_test = np.array(r_test).reshape(-1,512,1)
    lbls_test = np.array(lbls_test)
    # task_index_test = np.array(task_index_test)
    task_index_test = np.array(task_index_test).reshape(-1,1)

    scores_valid, scores_test, best_model = evaluate_model(args, train_ds, k=args.fold, shuffle=True, l_test=l_test, task_index_test=task_index_test, lbls_test=lbls_test)

    # Make predictions using the best_model
    best_score_test = {}
    test_r = best_model.evaluate([l_test,task_index_test], lbls_test, verbose=1)
    test_pred = best_model.predict([l_test,task_index_test])
    test_pred = np.array(test_pred).reshape(-1); test_pred_ = np.where(test_pred>0.5,1,0)
    lbls_test = lbls_test.reshape(-1)
    f1_test,mcc_test,bacc_test = compute_metrics(lbls_test, test_pred_)
    best_score_test['Acc'] = test_r[1]; best_score_test['AUC'] = test_r[2]; best_score_test['F1'] = f1_test; best_score_test['MCC'] = mcc_test; best_score_test['BACC'] = bacc_test;
    best_score_test['Precision'] = test_r[3]; best_score_test['Recall'] = test_r[4]
    result_df = pd.DataFrame(data={'smiles':smiles_test,'task_name':task_name_list,'test_pred':test_pred,'label_true':lbls_test})
    
    return best_score_test, best_model, scores_valid, scores_test, result_df
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', type=str, default='/home/datahouse1/liujin/HetSia-SafeNet/Data/ADR_multitask_dataset_scaffold_split') #ADR_multitask_dataset_random_split
    parser.add_argument('--dataset_path', type=str, default='/home/datahouse1/liujin/HetSia-SafeNet/Data/')
    parser.add_argument('--dataset', type=str, default="ADR_multitask_dataset_scaffold_split") #ADR_multitask_dataset_scaffold_split
    parser.add_argument('--result_path', type=str, default='/home/datahouse1/liujin/HetSia-SafeNet/result_split')
    parser.add_argument('--model_feature_type', type=str, default="property_feature_adr_emb_scaffold", choices=["property_feature_adr_emb_random", "property_feature_adr_emb_scaffold"])
    parser.add_argument('--view_list', type=str, default="AC50,Cmax", choices=["AC50,Cmax",])
    parser.add_argument('--device', type=str, default='cuda:1')
    # parser.add_argument('--device_id', type=int, default='2,3')
    parser.add_argument('--epoch_s', type=int, default=100)#30
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64) #64
    parser.add_argument('--lr', type=float, default=0.0001) #0.0001

    args = parser.parse_args()

    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_X = pd.read_csv(os.path.join(args.dataset_path, args.dataset,"AC50_Cmax","train_X.csv"))
    val_X = pd.read_csv(os.path.join(args.dataset_path, args.dataset,"AC50_Cmax","valid_X.csv"))
    test_X = pd.read_csv(os.path.join(args.dataset_path, args.dataset,"AC50_Cmax","test_X.csv"))
    train_y = pd.read_csv(os.path.join(args.dataset_path, args.dataset,"AC50_Cmax","train_y.csv"))
    val_y = pd.read_csv(os.path.join(args.dataset_path, args.dataset,"AC50_Cmax","valid_y.csv"))
    test_y = pd.read_csv(os.path.join(args.dataset_path, args.dataset,"AC50_Cmax","test_y.csv"))
    model_save_path = os.path.join(args.result_path, args.model_feature_type, 'AC50_Cmax')
    pathlib.Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
   
    sider_tasks = train_y.columns.values[1:].tolist()
    
    best_score_test, best_model, scores_valid, scores_test, result_df = model_train_and_eva(args, train_X, train_y, test_X, test_y, val_X, val_y, sider_tasks)
    result_df.to_csv(os.path.join(model_save_path, 'test_pred_result.csv'), index=False)
    best_model.save(os.path.join(model_save_path, 'best_model'), save_format='tf')

    # Save the model weight
    best_model.save_weights(os.path.join(model_save_path, 'best_model_weights'), save_format='tf')
  
    # Write best_score_test, scores_valid and scores_test into the file
    with open(os.path.join(model_save_path, 'test_metrics_result.txt'), 'w') as f:
        f.write('best_score_test: {}\n'.format(best_score_test))
        f.write('scores_valid: {}\n'.format(scores_valid))
        f.write('scores_test: {}\n'.format(scores_test))

    # Write scores_valid and scores_test into the csv file.
    scores_valid_df = pd.DataFrame(scores_valid, index=[i for i in range(args.fold)])
    scores_valid_df.to_csv(os.path.join(model_save_path, 'scores_valid.csv'))
    scores_test_df = pd.DataFrame(scores_test, index=[i for i in range(args.fold)])
    scores_test_df.to_csv(os.path.join(model_save_path, 'scores_test.csv'))


# nohup python -u train_adr_property_adremb.py --view_list AC50,Cmax > train_adr_property_adremb_ac50_cmax_random.txt 2>&1 &
# nohup python -u train_adr_property_adremb.py --view_list AC50,Cmax > train_adr_property_adremb_ac50_cmax_scaffold.txt 2>&1 &
