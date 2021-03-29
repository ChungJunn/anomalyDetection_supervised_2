import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def eval_main(model, validiter, device, neptune=None):
    model.eval()

    preds = []
    targets = []
    clf_hidden = None

    # forward the whole dataset and obtain result
    for li, (anno, ys, end_of_data) in enumerate(validiter):
        anno = anno.to(dtype=torch.float32, device=device)
        ys = ys.to(dtype=torch.int64, device=device)

        #outs, clf_hidden = model(anno, clf_hidden)
        outs = model(anno)

        outs = outs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy().reshape(-1,1)

        preds.append(outs)
        targets.append(ys)

        if end_of_data == 1: break

    # obtain results using metrics
    preds = np.vstack(preds)
    targets = np.vstack(targets)

    preds = np.argmax(preds, axis=1)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    print(acc, prec, rec, f1)
    model.train()

    return acc, prec, rec, f1
