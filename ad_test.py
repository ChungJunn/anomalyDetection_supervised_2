import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from ad_data import AD_SUP2_RNN_ITERATOR2

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
    clf_hidden = None

    # forward the whole dataset and obtain result
    for li, (anno, ys) in enumerate(dataiter):
        anno = anno.to(dtype=torch.float32, device=device)
        ys = ys.to(dtype=torch.int64, device=device)

        #outs, clf_hidden = model(anno, clf_hidden)
        outs = model(anno)

        outs = outs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy().reshape(-1,1)

        preds.append(outs)
        targets.append(ys)

    # obtain results using metrics
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    preds = np.argmax(preds, axis=1)

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

    # forward the whole dataset and obtain result
    for li, (x_data, y_data) in enumerate(dataiter):
        x_data = x_data.to(dtype=torch.float32, device=device)
        y_data = y_data.to(dtype=torch.int64, device=device)

        output = model(x_data)

        valid_loss += float(criterion(output, y_data).detach())
        # valid_loss += criterion(output, y_data)

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



