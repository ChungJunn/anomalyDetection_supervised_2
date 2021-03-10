import pandas as pd
import numpy as np
import pickle as pkl
import os
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

def remove_column(data, header):
    use_cols=[]
    for i in range(data.shape[1]):
        if np.std(data[:,i]) == 0:
            header.pop(i)
            pass
        else:
            use_cols.append(data[:,i].reshape(-1,1))

    new_data = np.hstack(use_cols)
    print('old shape: ', data.shape)
    print('new shape: ', new_data.shape)
    return new_data, header

HOME = os.environ['HOME']
data_name = 'cnsm_exp1_data'
data_file = data_name + '.csv'
dir = '21.03.10-gnn_data'

data_path = HOME + '/' + 'autoregressor/data/raw/' + data_file
save_dir = HOME + '/' + 'autoregressor/data/' + data_name + '/' + dir + '/'
rnn_len = 16
stride = 1

print(data_path)
df_data = pd.read_csv(data_path)
df_data = df_data.drop(columns=['Total_Label','rcl_label'])

label_col = 'SLA_Label'
header = df_data.columns

label = (np.array(df_data)[:,-1]).reshape(-1,1).astype(np.int32)
data = np.array(df_data)[:,:-1]
avgs = np.mean(data, axis=0).reshape(1,-1)
stds = (np.std(data, axis=0) + 0.001).reshape(1,-1)

data -= avgs
data /= stds

data = np.hstack((data, label))

df_data = pd.DataFrame(data, columns=header)

# split into nodes
label_data = df_data.iloc[:, -1]
label_data = np.array(label_data).reshape(-1,1)

fw_data = df_data.iloc[:, 0:23]
fw_header = list(fw_data.columns)
fw_data = np.hstack((np.array(fw_data), label_data))
fw_header.append(label_col)

ids_data = df_data.iloc[:, 23:46]
ids_header = list(ids_data.columns)
ids_data = np.hstack((np.array(ids_data), label_data))
ids_header.append(label_col)

flowmon_data = df_data.iloc[:, 46:69]
flowmon_header = list(flowmon_data.columns)
flowmon_data = np.hstack((np.array(flowmon_data), label_data))
flowmon_header.append(label_col)

dpi_data = df_data.iloc[:, 69:92]
dpi_header = list(dpi_data.columns)
dpi_data = np.hstack((np.array(dpi_data), label_data))
dpi_header.append(label_col)

lb_data = df_data.iloc[:, 92:115]
lb_header = list(lb_data.columns)
lb_data = np.hstack((np.array(lb_data), label_data))
lb_header.append(label_col)

# extract sequence first
fw_ns, fw_ans = extract_seqs(fw_data, rnn_len, stride)
ids_ns, ids_ans = extract_seqs(ids_data, rnn_len, stride)
flowmon_ns, flowmon_ans = extract_seqs(flowmon_data, rnn_len, stride)
dpi_ns, dpi_ans = extract_seqs(dpi_data, rnn_len, stride)
lb_ns, lb_ans = extract_seqs(lb_data, rnn_len, stride)

# index-based shuffling
# out : splited datasets
# train, valid, test(~9:30)
n_ns = len(fw_ns)
n_ans = len(fw_ans)

tr_ratio = 0.8
val_ratio = 0.1

ns_val_idx = int(n_ns * tr_ratio)
ans_val_idx = int(n_ans * tr_ratio)

ns_test_idx = int(n_ns * (tr_ratio + val_ratio))
ans_test_idx = int(n_ans * (tr_ratio + val_ratio))

fw_tr = np.vstack((fw_ns[:ns_val_idx,:,:-1], fw_ans[:ns_val_idx,:,:-1]))
fw_val = np.vstack((fw_ns[ns_val_idx:ns_test_idx,:,:-1], fw_ans[ns_val_idx:ns_test_idx,:,:-1]))
fw_test = np.vstack((fw_ns[ns_test_idx:,:,:-1], fw_ans[ns_test_idx:,:,:-1]))

ids_tr = np.vstack((ids_ns[:ns_val_idx,:,:-1], ids_ans[:ns_val_idx,:,:-1]))
ids_val = np.vstack((ids_ns[ns_val_idx:ns_test_idx,:,:-1], ids_ans[ns_val_idx:ns_test_idx,:,:-1]))
ids_test = np.vstack((ids_ns[ns_test_idx:,:,:-1], ids_ans[ns_test_idx:,:,:-1]))

flowmon_tr = np.vstack((flowmon_ns[:ns_val_idx,:,:-1], flowmon_ans[:ns_val_idx,:,:-1]))
flowmon_val = np.vstack((flowmon_ns[ns_val_idx:ns_test_idx,:,:-1], flowmon_ans[ns_val_idx:ns_test_idx,:,:-1]))
flowmon_test = np.vstack((flowmon_ns[ns_test_idx:,:,:-1], flowmon_ans[ns_test_idx:,:,:-1]))

dpi_tr = np.vstack((dpi_ns[:ns_val_idx,:,:-1], dpi_ans[:ns_val_idx,:,:-1]))
dpi_val = np.vstack((dpi_ns[ns_val_idx:ns_test_idx,:,:-1], dpi_ans[ns_val_idx:ns_test_idx,:,:-1]))
dpi_test = np.vstack((dpi_ns[ns_test_idx:,:,:-1], dpi_ans[ns_test_idx:,:,:-1]))

lb_tr = np.vstack((lb_ns[:ns_val_idx,:,:-1], lb_ans[:ns_val_idx,:,:-1]))
lb_val = np.vstack((lb_ns[ns_val_idx:ns_test_idx,:,:-1], lb_ans[ns_val_idx:ns_test_idx,:,:-1]))
lb_test = np.vstack((lb_ns[ns_test_idx:,:,:-1], lb_ans[ns_test_idx:,:,:-1]))

label_tr = np.vstack((lb_ns[:ns_val_idx,-1,-1:], lb_ans[:ns_val_idx,-1,-1:]))
label_val = np.vstack((lb_ns[ns_val_idx:ns_test_idx,-1,-1:], lb_ans[ns_val_idx:ns_test_idx,-1,-1:]))
label_test = np.vstack((lb_ns[ns_test_idx:,-1,-1:], lb_ans[ns_test_idx:,-1,-1:]))

# save into pkl files
#fw_tr.rnn_len00.stride00.pkl
with open(save_dir + 'sup_train.rnn_len' + str(rnn_len) +  '.fw.pkl', 'wb') as fp:
    pkl.dump(fw_tr, fp)
with open(save_dir + 'sup_val.rnn_len' + str(rnn_len) + '.fw.pkl', 'wb') as fp:
    pkl.dump(fw_val, fp)
with open(save_dir + 'sup_test.rnn_len' + str(rnn_len) + '.fw.pkl', 'wb') as fp:
    pkl.dump(fw_test, fp)

with open(save_dir + 'sup_train.rnn_len' + str(rnn_len) + '.ids.pkl', 'wb') as fp:
    pkl.dump(ids_tr, fp)
with open(save_dir + 'sup_val.rnn_len' + str(rnn_len) + '.ids.pkl', 'wb') as fp:
    pkl.dump(ids_val, fp)
with open(save_dir + 'sup_test.rnn_len' + str(rnn_len) + '.ids.pkl', 'wb') as fp:
    pkl.dump(ids_test, fp)

with open(save_dir + 'sup_train.rnn_len' + str(rnn_len) + '.flowmon.pkl', 'wb') as fp:
    pkl.dump(flowmon_tr, fp)
with open(save_dir + 'sup_val.rnn_len' + str(rnn_len) + '.flowmon.pkl', 'wb') as fp:
    pkl.dump(flowmon_val, fp)
with open(save_dir + 'sup_test.rnn_len' + str(rnn_len) + '.flowmon.pkl', 'wb') as fp:
    pkl.dump(flowmon_test, fp)

with open(save_dir + 'sup_train.rnn_len' + str(rnn_len) + '.dpi.pkl', 'wb') as fp:
    pkl.dump(dpi_tr, fp)
with open(save_dir + 'sup_val_rnn_len' + str(rnn_len) + '.dpi.pkl', 'wb') as fp:
    pkl.dump(dpi_val, fp)
with open(save_dir + 'sup_test.rnn_len' + str(rnn_len) + '.dpi.pkl', 'wb') as fp:
    pkl.dump(dpi_test, fp)

with open(save_dir + 'sup_train.rnn_len' + str(rnn_len) + '.lb.pkl', 'wb') as fp:
    pkl.dump(lb_tr, fp)
with open(save_dir + 'sup_val.rnn_len' + str(rnn_len) + '.lb.pkl', 'wb') as fp:
    pkl.dump(lb_val, fp)
with open(save_dir + 'sup_test.rnn_len' + str(rnn_len) + '.lb.pkl', 'wb') as fp:
    pkl.dump(lb_test, fp)

with open(save_dir + 'sup_train.rnn_len' + str(rnn_len) + '.label.pkl', 'wb') as fp:
    pkl.dump(label_tr, fp)
with open(save_dir + 'sup_val.rnn_len' + str(rnn_len) + '.label.pkl', 'wb') as fp:
    pkl.dump(label_val, fp)
with open(save_dir + 'sup_test.rnn_len' + str(rnn_len) + '.label.pkl', 'wb') as fp:
    pkl.dump(label_test, fp)
