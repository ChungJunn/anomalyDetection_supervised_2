import numpy as np
import pandas as pd
import argparse

# 1. implement switch - label:'rcl', 'sla' - argparse/shell
# 2. implement encoding code
# 3. configure encoding files
# 4. test

def prepare_csv_file(csv_path, label, dict_dir=None):
    if label == 'sla':
        li=-3 # use sla label
    elif label == 'rcl' or label == 'rcl-no_hop':
        li=-1
        if dict_dir is None:
            print("must provide dict dir for encoding rcl")
            import sys; sys.exit(0)
    else:
        print("label must be either sla or rcl")
        import sys; sys.exit(0)

    # import data
    data = pd.read_csv(csv_path)
    header = list(data.columns)
    data = np.array(data) # 95 cols

    # divide input and label
    xs = data[:,:-3] # 92 cols
    input_header = header[:-3]
    label_header = header[-3:]
    ys = data[:,-3:]

    # if rcl, encode
    if li==-1:
        d = {}
        with open(dict_dir, 'rt') as fp:
            lines = fp.readlines()
            for line in lines:
                ps = line.strip().split(',')
                d[ps[0]] = int(ps[1])

        for n in range(len(ys)):
            ys[n, -1] = d[ys[n, -1]]

    # concat input and label
    data = np.hstack((xs, ys[:,li].reshape(-1,1)))
    header = input_header + [label_header[li]]

    data_dim=data.shape[1]
    data_len=len(data)

    # if rcl-no_hop, then exclude hop-by-hop features
    if label == 'rcl-no_hop':
        idxlist = list(range(data_dim))
        for i in range(data_dim-1, 0, -1):
            if i > 0 and i % 23 == 22:
                header.pop(i)
                idxlist.pop(i)

        data = data[:, idxlist]

        data_dim=data.shape[1]
        data_len=len(data)

    # remove std==0 columns
    use_cols=[]
    for i in range(data_dim):
        if np.std(data[:,i]) == 0:
            header.pop(i)
            pass
        else:
            use_cols.append(data[:,i].reshape(-1,1))

    data = np.hstack(use_cols) # 88 cols + 1 col (label)
    data_dim = data.shape[1]

    header = ','.join(header)

    return data, header

def extract_seqs(data, rnn_len, stride):
    '''
    extracts seqs and classify to normal and abnormal seqs and save them in nparray and return
    args:
        data: nparray, (data_len, data_dim), VNF anomaly detection dataset (time series)
        rnn_len: int, length of sequence to extract
        stride: int, stride to use
    return:
        normal_seqs: nparray, (seq_n, rnn_len+1, data_dim)
        abnormal_seqs: nparray, (seq_n, rnn_len+1, data_dim)
    '''
    n_seqs = []
    a_seqs = []
    idx = 0

    for idx in range(0, len(data), stride):
        if (idx + rnn_len + 1) > len(data): break

        data_range = range(idx, idx+rnn_len+1)
        seq = data[data_range, :]

        if seq[-1, -1] != 0:
            a_seqs.append(seq)
        else:
            n_seqs.append(seq)

    ns = np.stack(n_seqs)
    ans = np.stack(a_seqs)

    return ns, ans

def split_dataset(normal_seqs, abnormal_seqs):
    '''
    extracts tr, val1, val2, and test from normals and abnormal sequences
    '''
    np.random.shuffle(normal_seqs)
    np.random.shuffle(abnormal_seqs)

    # extract test data
    sp = 0.8
    n_idx = int(np.ceil(len(normal_seqs) * sp))
    a_idx = int(np.ceil(len(abnormal_seqs) * sp))

    test = np.vstack([normal_seqs[n_idx:], abnormal_seqs[a_idx:]])
    normal_seqs = normal_seqs[:n_idx]
    abnormal_seqs = abnormal_seqs[:a_idx]

    # extract val2 data
    sp = 0.9
    n_idx = int(np.ceil(len(normal_seqs) * sp))
    a_idx = int(np.ceil(len(abnormal_seqs) * sp))

    val2 = np.vstack([normal_seqs[n_idx:], abnormal_seqs[a_idx:]])
    normal_seqs = normal_seqs[:n_idx]
    abnormal_seqs = abnormal_seqs[:a_idx]

    # extract val1 data
    sp = 0.9
    n_idx = int(np.ceil(len(normal_seqs) * sp))

    val1 = normal_seqs[n_idx:]
    train = normal_seqs[:n_idx]

    # shuffle
    np.random.shuffle(train)
    np.random.shuffle(val1)
    np.random.shuffle(val2)
    np.random.shuffle(test)
    np.random.shuffle(abnormal_seqs)

    # convert types
    train = train.astype(np.float32)
    val1 = val1.astype(np.float32)
    val2 = val2.astype(np.float32)
    test = test.astype(np.float32)
    abnormal_seqs = abnormal_seqs.astype(np.float32)

    return train, val1, val2, test, abnormal_seqs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='', default='')

    parser.add_argument('--rnn_len', type=int, help='', default=0)
    parser.add_argument('--stride', type=int, help='', default=0)
    parser.add_argument('--label', type=str, help='', default='')
    parser.add_argument('--dict_dir', type=str, help='', default='')

    parser.add_argument('--ar_tr_out', type=str, help='', default='')
    parser.add_argument('--ar_val1_out', type=str, help='', default='')
    parser.add_argument('--ar_val2_out', type=str, help='', default='')
    parser.add_argument('--ar_test_out', type=str, help='', default='')
    parser.add_argument('--ar_stat_file', type=str, help='', default='')

    parser.add_argument('--ae_tr_out', type=str, help='', default='')
    parser.add_argument('--ae_val1_out', type=str, help='', default='')
    parser.add_argument('--ae_val2_out', type=str, help='', default='')
    parser.add_argument('--ae_test_out', type=str, help='', default='')

    parser.add_argument('--suprnn_tr_out', type=str, help='', default='')
    parser.add_argument('--suprnn_val_out', type=str, help='', default='')
    parser.add_argument('--suprnn_test_out', type=str, help='', default='')

    parser.add_argument('--sup_tr_out', type=str, help='', default='')
    parser.add_argument('--sup_val_out', type=str, help='', default='')
    parser.add_argument('--sup_test_out', type=str, help='', default='')

    args = parser.parse_args()

    data, header = prepare_csv_file(args.csv_path, args.label, args.dict_dir)

    # autoregressor dataset
    ns, ans = extract_seqs(data, args.rnn_len, args.stride)

    train, val1, val2, test, remains = split_dataset(ns, ans)

    import pickle as pkl
    with open(args.ar_tr_out, 'wb') as fp:
        pkl.dump(train, fp)
    with open(args.ar_val1_out, 'wb') as fp:
        pkl.dump(val1, fp)
    with open(args.ar_val2_out, 'wb') as fp:
        pkl.dump(val2, fp)
    with open(args.ar_test_out, 'wb') as fp:
        pkl.dump(test, fp)

    # for supervised rnn dataset
    suprnn_train = np.vstack([train, val1, val2, remains])
    np.random.shuffle(suprnn_train)
    sp = 0.9
    sp_idx = int(np.ceil(len(suprnn_train) * sp))
    suprnn_val = suprnn_train[sp_idx:]
    suprnn_train = suprnn_train[:sp_idx]

    import pickle as pkl
    with open(args.suprnn_tr_out, 'wb') as fp:
        pkl.dump(suprnn_train, fp)
    with open(args.suprnn_val_out, 'wb') as fp:
        pkl.dump(suprnn_val, fp)
    with open(args.suprnn_test_out, 'wb') as fp:
        pkl.dump(test, fp)

    # output stats
    train = train[:,-1,:]
    val1 = val1[:,-1,:]
    val2 = val2[:,-1,:]
    test = test[:,-1,:]
    remain = remains[:,-1,:]

    # for autoencoder learning dataset
    np.savetxt(args.ae_tr_out, train, delimiter=',', header=header)
    np.savetxt(args.ae_val1_out, val1, delimiter=',', header=header)
    np.savetxt(args.ae_val2_out, val2, delimiter=',', header=header)
    np.savetxt(args.ae_test_out, test, delimiter=',', header=header)

    # compute stats except for label column
    # stats for total 88 columns
    arr = train[:,:-1]
    x_avg = np.mean(arr, axis=0)
    x_std = np.std(arr, axis=0)

    fp = open(args.ar_stat_file, 'w')
    for i in range(x_avg.shape[0]):
        if i > 0:
            fp.write(', ')
        fp.write('%.9f' % (x_avg[i]))
    fp.write('\n')

    for i in range(x_std.shape[0]):
        if i > 0:
            fp.write(', ')
        fp.write('%.9f' % (x_std[i]))
    fp.write('\n')
    fp.close()

    # for supervised learning dataset
    # reunite the data points, shuffle, and divide to train and valid
    data = np.vstack([train, val1, val2, remain])

    np.random.shuffle(data)

    sp = 0.9
    sp_idx = int(np.ceil(len(data) * sp))

    train = data[:sp_idx]
    val = data[sp_idx:]

    np.savetxt(args.sup_tr_out, train, delimiter=',', header=header)
    np.savetxt(args.sup_val_out, val, delimiter=',', header=header)
    np.savetxt(args.sup_test_out, test, delimiter=',', header=header)
