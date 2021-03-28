import yaml
import argparse
import pandas as pd
import csv
import os
import sys
import pandas as pd
from Feature_extract import feature_transform
from Datagenerator import Datagen
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import *
from tqdm import tqdm
from collections import Counter
from batch_sampler import EpisodicBatchSampler
from torch.nn import functional as F
from util import prototypical_loss as loss_fn
from util import evaluate_prototypes
from glob import glob
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text


def train_protonet(model,train_loader,valid_loader,conf,num_batches_tr,num_batches_vd):

    '''Model training
    Args:
    -model: Model
    -train_laoder: Training loader
    -valid_load: Valid loader
    -conf: configuration object
    -num_batches_tr: number of training batches
    -num_batches_vd: Number of validation batches

    Out:
    -best_val_acc: Best validation accuracy
    -model
    -best_state: State dictionary for the best validation accuracy
    '''

    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam(model.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    num_epochs = conf.train.epochs

    best_model_path = conf.path.best_model
    last_model_path = conf.path.last_model
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_acc = 0.0
    model.to(device)

    for epoch in range(num_epochs):

        print("Epoch {}".format(epoch))
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x_out = model(x)
            tr_loss,tr_acc = loss_fn(x_out,y,conf.train.n_shot)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())

            tr_loss.backward()
            optim.step()

        avg_loss_tr = np.mean(train_loss[-num_batches_tr:])
        avg_acc_tr = np.mean(train_acc[-num_batches_tr:])
        print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr,avg_acc_tr))
        lr_scheduler.step()
        model.eval()
        val_iterator = iter(valid_loader)

        for batch in tqdm(val_iterator):
            x,y = batch
            x = x.to(device)
            x_val = model(x)
            valid_loss, valid_acc = loss_fn(x_val, y, conf.train.n_shot)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc.item())
        avg_loss_vd = np.mean(val_loss[-num_batches_vd:])
        avg_acc_vd = np.mean(val_acc[-num_batches_vd:])

        print ('Epoch {}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(epoch,avg_loss_vd,avg_acc_vd))
        if avg_acc_vd > best_val_acc:
            print("Saving the best model with valdation accuracy {}".format(avg_acc_vd))
            best_val_acc = avg_acc_vd
            best_state = model.state_dict()
            torch.save(model.state_dict(),best_model_path)
    torch.save(model.state_dict(),last_model_path)

    return best_val_acc,model,best_state


def visualize_prototypes(model,train_loader,conf):

    '''Model training
    Args:
    -model: Model
    -train_laoder: Training loader
    -conf: configuration object

    Out:
    -dict
    '''

    device = torch.device('cuda')
    model.load_state_dict(torch.load(conf.path.best_model))
    model.to(device)
    model.eval()

    train_iterator = iter(train_loader)
    d = {}
    for batch in tqdm(train_iterator):
        x, y = batch
        x = x.to(device)
        x_out = model(x)
        y = y.numpy()
        for i,yi in enumerate(y):
            if int(yi) in d:
                d[int(yi)][0] += x_out[i].cpu().detach().numpy()
                d[int(yi)][1] += 1
            else:
                d[int(yi)] = [x_out[i].cpu().detach().numpy(), 1]
    d2 = {}
    for k in d.keys():
        d2[k] = d[k][0]/d[k][1]
    return d2


@hydra.main(config_name="config")
def main(conf : DictConfig):

    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)

    if not os.path.isdir(conf.path.feat_train):
        os.makedirs(conf.path.feat_train)

    if not os.path.isdir(conf.path.feat_eval):
        os.makedirs(conf.path.feat_eval)

    if conf.set.features:

        print(" --Feature Extraction Stage--")
        Num_extract_train,data_shape = feature_transform(conf=conf,mode="train")
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(Num_extract_train))

        Num_extract_eval = feature_transform(conf=conf,mode='eval')
        print("Total number of samples used for evaluation: {}".format(Num_extract_eval))
        print(" --Feature Extraction Complete--")

    Protonet = getattr(sys.modules[__name__], conf.model)

    if conf.set.train:

        if not os.path.isdir(conf.path.model):
            os.makedirs(conf.path.model)


        gen_train = Datagen(conf)
        X_train,Y_train,X_val,Y_val = gen_train.generate_train()
        X_tr = torch.tensor(X_train)
        Y_tr = torch.LongTensor(Y_train)
        X_val = torch.tensor(X_val)
        Y_val = torch.LongTensor(Y_val)

        samples_per_cls =  conf.train.n_shot * 2

        batch_size_tr = samples_per_cls * conf.train.k_way
        batch_size_vd = batch_size_tr

        num_batches_tr = len(Y_train)//batch_size_tr
        num_batches_vd = len(Y_val)//batch_size_vd


        samplr_train = EpisodicBatchSampler(Y_train,num_batches_tr,conf.train.k_way,samples_per_cls)
        samplr_valid = EpisodicBatchSampler(Y_val,num_batches_vd,conf.train.k_way,samples_per_cls)

        train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr)
        valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,num_workers=8,pin_memory=True,shuffle=False)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,num_workers=8,pin_memory=True,shuffle=False)

        model = Protonet()
        best_acc,model,best_state = train_protonet(model,train_loader,valid_loader,conf,num_batches_tr,num_batches_vd)
        print("Best accuracy of the model on training set is {}".format(best_acc))

    if conf.set.viz:

        if not os.path.isdir(conf.path.model):
            print("No model found")
            return

        gen_train = Datagen(conf)
        X_train,Y_train,X_val,Y_val = gen_train.generate_train()
        X_tr = torch.tensor(X_train)
        Y_tr = torch.LongTensor(Y_train)

        samples_per_cls =  conf.train.n_shot * 2
        batch_size_tr = samples_per_cls * conf.train.k_way
        num_batches_tr = len(Y_train)//batch_size_tr
        samplr_train = EpisodicBatchSampler(Y_train,num_batches_tr,conf.train.k_way,samples_per_cls)
        train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,num_workers=8,pin_memory=True,shuffle=False)

        model = Protonet()
        d = visualize_prototypes(model,train_loader,conf)

        pca = PCA(n_components=2)
        m = np.max(list(d.keys()))
        dim = d[0].size
        X = np.zeros((m+1,dim))
        for i in range(m+1):
            X[i] = d[i]
        X = pca.fit_transform(X)

        plt.axis('off')
        plt.title(conf.model + " Class Prototypes")

        texts = []
        for i in range(len(X)):
            l = gen_train.idx2label[i]
            if l in ["AMRE","BBWA","BTBW","COYE", "CHSP", "GCTH", "OVEN","RBGR", "SAVS", "SWTH", "WTSP"]:
                texts += [plt.text(X[i][0], X[i][1], l, c="black")]
                plt.scatter([X[i][0]], [X[i][1]], c="black", s=12)
            elif l in ["JD"]:
                texts += [plt.text(X[i][0], X[i][1], l, c="blue")]
                plt.scatter([X[i][0]], [X[i][1]], c="blue", s=12)
            elif l in ["SQT", "GRN", "GIG"]:
                texts += [plt.text(X[i][0], X[i][1], l, c="green")]
                plt.scatter([X[i][0]], [X[i][1]], c="green", s=12)
            elif l in ["AGGM", "SNMK", "CCMK", "SOCM"]:
                texts += [plt.text(X[i][0], X[i][1], l, c="purple")]
                plt.scatter([X[i][0]], [X[i][1]], c="purple", s=12)
            else:
                texts += [plt.text(X[i][0], X[i][1], l, c="blue")]
                plt.scatter([X[i][0]], [X[i][1]], c="blue", s=12)

        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), expand_text=(1.5,1.5))
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='BirdVox',
                               markerfacecolor='black', markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Jackdaw',
                                markerfacecolor='blue', markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Hyena',
                                markerfacecolor='green', markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Meerkat',
                                markerfacecolor='purple', markersize=8),]
        lgd = plt.legend(handles=legend_elements, bbox_to_anchor=(1.2,1))
        plt.savefig(conf.path.root_dir + "/centroids.png", bbox_extra_artists=(lgd,), bbox_inches='tight')

    if conf.set.eval:

        device = 'cuda'

        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = [file for file in glob(os.path.join(conf.path.feat_eval,'*.h5'))]

        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')

            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            model = Protonet()
            onset,offset = evaluate_prototypes(model, conf,hdf_eval,device,strt_index_query)

            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        csv_path = os.path.join(conf.path.root_dir,'Eval_out.csv')
        df_out.to_csv(csv_path,index=False)


if __name__ == '__main__':
     main()

