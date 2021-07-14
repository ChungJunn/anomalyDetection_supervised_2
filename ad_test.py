import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def eval_joint_forward(model, dataiter, device):
    model.eval()

    preds = []
    targets = []
    clf_hidden = None

    # forward the whole dataset and obtain result
    for li, (xs, ys, end_of_data) in enumerate(dataiter):
        xs = xs.to(dtype=torch.float32, device=device)
        ys = ys.to(dtype=torch.int64, device=device)

        #outs, clf_hidden = model(anno, clf_hidden)
        outs = model(xs)

        outs = outs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy().reshape(-1,1)

        preds.append(outs)
        targets.append(ys)

        if end_of_data == 1:
            break

    # obtain results using metrics
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    preds = np.argmax(preds, axis=1)

    model.train()

    return targets, preds

def eval_forward(model, dataiter, device):
    model.eval()

    preds = []
    targets = []

    # forward the whole dataset and obtain result
    for li, (x_data, y_data) in enumerate(dataiter):
        ####
        Bn, Tx, V, D = x_data.size()
        x_data = x_data.to(dtype=torch.float32, device=device) # Bn Tx V D
        y_data = y_data.to(dtype=torch.int64, device=device) # Bn Tx 1
        y_prev = torch.zeros(Bn,1).to(dtype=torch.int64, device=device) # Bn 1
        clf_hidden = model.init_clf_hidden(Bn, device)
        
        for di in range(Tx):
            x_t = x_data[:,di,:,:]
            y_prev = y_prev.unsqueeze(1).expand(-1,V,-1).contiguous()

            input_data = torch.cat((x_t, y_prev), dim=-1).unsqueeze(1)
            
            output, clf_hidden = model(input_data, clf_hidden)

            topv, topi = output.topk(1)
            y_prev = topi.detach()  # detach from history as input

            if di == (Tx - 1):
                preds.append(topi.detach().cpu().numpy())
                targets.append(y_data[:, di].unsqueeze(-1).cpu().numpy())
        ####
        '''
        #outs, clf_hidden = model(anno, clf_hidden)
        outs = model(x_data)

        outs = outs.detach().cpu().numpy()
        y_data = y_data.detach().cpu().numpy().reshape(-1,1)

        preds.append(outs)
        targets.append(y_data)
        '''

    # obtain results using metrics
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    model.train()

    return targets, preds

def eval_binary(targets, preds):
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    return acc, prec, rec, f1

def log_neptune(result_dict, mode, neptune): # mode: valid, best_valid, test
    for class_item in result_dict.keys():
            if class_item == "accuracy":
                prefix = mode + " accuracy"
                score = result_dict[class_item]
                neptune.set_property(prefix, score)

            else:
                for metric in result_dict[class_item].keys():
                    prefix = mode + ' ' + class_item + ' ' + metric
                    score = (result_dict[class_item])[metric]
                    neptune.set_property(prefix, score)

def get_combined_valid_loss(model, dataiter1, dataiter2, criterion, device):
    model.eval()
    valid_loss = 0.0
    n_batches = 0

    for li, ((x_data1, y_data1, end_of_data1), (x_data2, y_data2, end_of_data2)) in enumerate(zip(dataiter1, dataiter2)):
        x_data1, x_data2 = x_data1.to(dtype=torch.float32, device=device), x_data2.to(dtype=torch.float32, device=device)
        y_data1, y_data2 = y_data1.to(dtype=torch.int64, device=device), y_data2.to(dtype=torch.int64, device=device) 

        output1 = model(x_data1)
        output2 = model(x_data2)

        # go through loss function
        loss1 = float(criterion(output1, y_data1).detach())
        loss2 = float(criterion(output2, y_data2).detach())
        valid_loss += loss1 + loss2

        if end_of_data2 == 1: break

    valid_loss /= (li + 1)

    model.train()

    return valid_loss

def get_valid_loss(model, dataiter, criterion, device):
    model.eval()
    valid_loss = 0.0
    loss = 0

    # forward the whole dataset and obtain result
    for li, (x_data, y_data) in enumerate(dataiter):
        Bn, Tx, V, D = x_data.size()
        x_data = x_data.to(dtype=torch.float32, device=device) # Bn Tx V D
        y_data = y_data.to(dtype=torch.int64, device=device) # Bn Tx 1
        y_prev = torch.zeros(Bn,1).to(dtype=torch.int64, device=device) # Bn 1
        clf_hidden = model.init_clf_hidden(Bn, device)
        
        for di in range(Tx):
            x_t = x_data[:,di,:,:]
            y_prev = y_prev.unsqueeze(1).expand(-1,V,-1).contiguous()

            input_data = torch.cat((x_t, y_prev), dim=-1).unsqueeze(1)
            
            output, clf_hidden = model(input_data, clf_hidden)

            loss += float(criterion(output, y_data[:, di]))

            topv, topi = output.topk(1)
            y_prev = topi.detach()  # detach from history as input

        valid_loss += (loss / (di + 1))
        loss = 0
        
    # obtain results using metrics
    valid_loss /= (li + 1)

    model.train()

    return valid_loss

if __name__ == '__main__':
    my_dict = {'label 1': {'precision':0.5,
                'recall':1.0,
                'f1-score':0.67},
              'label 2': {'precision':0.2,
                'recall':0.2,
                'f1-score':0.2}
    }



