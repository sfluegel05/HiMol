import argparse
import os
from cmath import inf

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from loader import DEEPCHEM_MOLNET_DATASETS, MoleculeDataset
from metrics import MacroF1
from model import GNN, GNN_graphpred
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from splitters import chebi_split, random_split, scaffold_split
from torch_geometric.loader import DataLoader
from torchmetrics.classification import MultilabelF1Score
from tqdm import tqdm

criterion = nn.BCEWithLogitsLoss(reduction = "none")


def print_parameter_summary(module, title):
    print(f"\n{title}")
    print("-" * len(title))
    total_params = 0
    for name, param in module.named_parameters():
        total_params += param.numel()
        print(f"  {name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")
    print(f"Total parameters: {total_params:,}")


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def train_reg(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        if args.dataset in ['qm7', 'qm8', 'qm9']:
            loss = torch.sum(torch.abs(pred-y))/y.size(0)
        elif args.dataset in ['esol','freesolv','lipophilicity']:
            loss = torch.sum((pred-y)**2)/y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    #Whether y is non-null or not.
    y = batch.y.view(pred.shape).to(torch.float64)
    is_valid = y**2 > 0
    #Loss matrix
    loss_mat = criterion(pred.double(), (y+1)/2)
    #loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    loss = torch.sum(loss_mat)/torch.sum(is_valid)


    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    eval_roc = sum(roc_list)/len(roc_list) #y_true.shape[1]

    return eval_roc, loss

def eval_chebi(args, model, device, loader):
    """Multi-label evaluation for ChEBI: mean per-task ROC-AUC plus micro/macro
    F1 (the standard chebai metric). Returns a dict of metrics."""
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()      # -1 / +1
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()  # logits

    # loss over all valid (non-missing) entries; ChEBI has no missing labels
    is_valid = y_true ** 2 > 0
    loss_mat = criterion(torch.tensor(y_scores).double(),
                         torch.tensor((y_true + 1) / 2).double())
    loss_mat = torch.where(torch.tensor(is_valid), loss_mat,
                           torch.zeros_like(loss_mat))
    loss = float(torch.sum(loss_mat) / max(int(is_valid.sum()), 1))

    # per-task ROC-AUC, skipping tasks without both classes present
    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            valid_i = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[valid_i, i] + 1) / 2,
                                          y_scores[valid_i, i]))
    eval_roc = sum(roc_list) / len(roc_list) if roc_list else 0.0

    # micro / macro F1 at logit threshold 0
    y_true_bin = (y_true + 1) / 2          # 0 / 1
    y_pred_bin = (y_scores > 0).astype(int)
    # micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    # macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    micro_f1_obj = MultilabelF1Score(num_labels=y_true.shape[1], average="macro")
    macro_f1_obj = MacroF1(num_labels=y_true.shape[1])

    micro_f1 = micro_f1_obj(torch.tensor(y_pred_bin), torch.tensor(y_true_bin))
    macro_f1 = macro_f1_obj(torch.tensor(y_pred_bin), torch.tensor(y_true_bin))

    return {'auc': eval_roc, 'micro_f1': micro_f1, 'macro_f1': macro_f1,
            'loss': loss}


def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse=np.sqrt(mean_squared_error(y_true,y_scores))
    return mse, mae, rmse

def save_emb(args, model, device, loader, num_tasks, out_file):
    model.eval()

    emb,label = [],[]
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        graph_emb = model.graph_emb(batch.x, batch.edge_index, batch.edge_attr, batch.batch).cpu().detach().numpy()
        y = batch.y.view(-1, num_tasks).cpu().detach().numpy()
        emb.append(graph_emb)
        label.append(y)
    output_emb = np.row_stack(emb)
    output_label = np.row_stack(label)

    np.savez(out_file, emb=output_emb, label=output_label)

       


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr_feat', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_pred', type=float, default=0.001,
                        help='learning rate for the prediction layer (default: 0.0005)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 512)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help='gnn_type (gat, gin, gcn, graphsage)')
    parser.add_argument('--dataset', type=str, default = 'chebi', 
                        help='[chebi, bbbp, bace, sider, clintox, sider,tox21, toxcast, esol,freesolv,lipophilicity]')
    parser.add_argument('--input_model_file', type=str, default = '../saved_model/pretrain.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--split_file', type=str, default='',
                        help='python-chebai split CSV (id,split); required for --dataset chebi')
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--GNN_para', type=bool, default = True, help='if the parameter of pretrain update')
    parser.add_argument('--wandb_entity', type=str, default='chebai',
                        help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb_project', type=str, default='chebai',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name (default: dataset-seed)')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode (use "disabled" to turn off logging)')
    args = parser.parse_args()

    run_name = args.wandb_run_name or ('%s-run%d' % (args.dataset, args.runseed))
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                     mode=args.wandb_mode, config=vars(args), tags=[args.dataset, 'himol'])

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'hiv', 'pcba', 'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'mutag', 'chebi']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == 'esol':
        num_tasks = 1
    elif args.dataset == 'freesolv':
        num_tasks = 1
    elif args.dataset == 'lipophilicity':
        num_tasks = 1
    elif args.dataset == 'qm7':
        num_tasks = 1
    elif args.dataset == 'qm8':
        num_tasks = 12
    elif args.dataset == 'qm9':
        num_tasks = 12
    elif args.dataset == 'chebi':
        # one task per ChEBI class, derived from the raw classes.txt
        with open('dataset/chebi/raw/classes.txt', encoding='utf-8') as f:
            num_tasks = sum(1 for line in f if line.strip() != '')
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    
    print(dataset)
    
    if args.dataset == 'chebi':
        print("Using ChEBI dataset")
        if args.split_file == '':
            raise ValueError("--split_file is required for --dataset chebi")
        ids_list = pd.read_csv('dataset/chebi/processed/chebi_ids.csv',
                               header=None, dtype=str)[0].tolist()
        train_dataset, valid_dataset, test_dataset = chebi_split(
            dataset, ids_list, args.split_file)
        print("chebi split from %s" % args.split_file)
    if args.dataset in DEEPCHEM_MOLNET_DATASETS:
        train_idx, valid_idx, test_idx = [], [], []
        for i, data in enumerate(dataset):
            fold = data.fold.item()
            if fold == 0:
                train_idx.append(i)
            elif fold == 1:
                valid_idx.append(i)
            elif fold == 2:
                test_idx.append(i)
        train_dataset = dataset[torch.tensor(train_idx)]
        valid_dataset = dataset[torch.tensor(valid_idx)]
        test_dataset = dataset[torch.tensor(test_idx)]
        print("deepchem molnet split")
    elif args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    if args.dataset == 'freesolv':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    print_parameter_summary(model.gnn, "GNN model parameters")
    print_parameter_summary(model.graph_pred_linear, "Classification head parameters")
    head_out_features = getattr(model.graph_pred_linear, "out_features", None)
    if head_out_features is not None:
        print(f"Classification head output size: {head_out_features}")

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    if args.GNN_para:
        print('GNN update')
        model_param_group.append({"params": model.gnn.parameters(), "lr":args.lr_feat})
    else:
        print('No GNN update')
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr_pred})
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    print(optimizer)

    finetune_model_save_path = './model_checkpoints/' + args.dataset + '.pth'
    os.makedirs(os.path.dirname(finetune_model_save_path), exist_ok=True)

   
    # training based on task type
    if task_type == 'cls' and args.dataset == 'chebi':
        # multi-label: report both micro/macro F1 (chebai standard) and mean AUC
        for epoch in range(1, args.epochs + 1):
            print('====epoch:', epoch)

            train(model, device, train_loader, optimizer)

            print('====Evaluation')
            log = {'epoch': epoch}
            if args.eval_train:
                train_m = eval_chebi(args, model, device, train_loader)
                for k, v in train_m.items():
                    log['train/' + k] = v
            else:
                print('omit the training accuracy computation')
            val_m = eval_chebi(args, model, device, val_loader)
            test_m = eval_chebi(args, model, device, test_loader)
            for k, v in val_m.items():
                log['val/' + k] = v
            for k, v in test_m.items():
                log['test/' + k] = v

            torch.save(model.state_dict(), finetune_model_save_path)

            print("val   micro_f1: %f macro_f1: %f auc: %f" %
                  (val_m['micro_f1'], val_m['macro_f1'], val_m['auc']))
            print("test  micro_f1: %f macro_f1: %f auc: %f" %
                  (test_m['micro_f1'], test_m['macro_f1'], test_m['auc']))

            run.log(log)

    elif task_type == 'cls':
        best_epoch_val = -1.0
        best_epoch_number = 0
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)

            train(model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_auc, train_loss = eval(args, model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_auc = 0
            val_auc, val_loss = eval(args, model, device, val_loader)

            if val_auc > best_epoch_val:  
                best_epoch_val = val_auc
                best_epoch_number = epoch
                torch.save(model.state_dict(), finetune_model_save_path)

            print("train_auc: %f val_auc: %f" %(train_auc, val_auc))

            run.log({'epoch': epoch,
                     'train/auc': train_auc, 'val/auc': val_auc,
                     'train/loss': float(train_loss), 'val/loss': float(val_loss),
                     })
                
        model.load_state_dict(torch.load(finetune_model_save_path))
        test_auc, test_loss = eval(args, model, device, test_loader)
        print("best epoch %d val_auc: %f test_auc: %f" %(best_epoch_number, best_epoch_val, test_auc))



    elif task_type == 'reg':
        train_list, test_list = [], []
        for epoch in range(1, args.epochs+1):
            print('====epoch:',epoch)
            
            train_reg(args, model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_mse, train_mae, train_rmse = eval_reg(args, model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_mse, train_mae, train_rmse = 0, 0, 0
            val_mse, val_mae, val_rmse = eval_reg(args, model, device, val_loader)
            test_mse, test_mae, test_rmse = eval_reg(args, model, device, test_loader)
            
            if args.dataset in ['esol', 'freesolv', 'lipophilicity']:
                test_list.append(float('{:.6f}'.format(test_rmse)))
                train_list.append(float('{:.6f}'.format(train_rmse)))
                torch.save(model.state_dict(), finetune_model_save_path)

            elif args.dataset in ['qm7', 'qm8', 'qm9']:
                test_list.append(float('{:.6f}'.format(test_mae)))
                train_list.append(float('{:.6f}'.format(train_mae)))
                torch.save(model.state_dict(), finetune_model_save_path)
                
            print("train_mse: %f val_mse: %f test_mse: %f" %(train_mse, val_mse, test_mse))
            print("train_mae: %f val_mae: %f test_mae: %f" %(train_mae, val_mae, test_mae))
            print("train_rmse: %f val_rmse: %f test_rmse: %f" %(train_rmse, val_rmse, test_rmse))

            run.log({'epoch': epoch,
                     'train/mse': train_mse, 'val/mse': val_mse, 'test/mse': test_mse,
                     'train/mae': train_mae, 'val/mae': val_mae, 'test/mae': test_mae,
                     'train/rmse': train_rmse, 'val/rmse': val_rmse, 'test/rmse': test_rmse})

    run.finish()



if __name__ == "__main__":
    main()
