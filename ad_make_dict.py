import numpy
import pickle as pkl

# make idx2class and class2idx
target_base_dir = '/home/chl/autoregressor/data/'
wsd_dict_path = target_base_dir + 'cnsm_exp1_data/' + 'dict.pkl'
lad1_dict_path = target_base_dir + 'cnsm_exp2_1_data/' + 'dict.pkl'
lad2_dict_path = target_base_dir + 'cnsm_exp2_2_data/' + 'dict.pkl'

# for wsd
class2idx = {'normal':0, 'fw':1, 'ids':2, 'fm':3, 'dpi':4, 'lb':5, 'none':6}
idx2class= ['normal', 'fw', 'ids', 'fm', 'dpi', 'lb', 'none']
dict = {'class2idx':class2idx, 'idx2class':idx2class}
with open(wsd_dict_path, 'wb') as fp:
    pkl.dump(dict, fp)

# for lad
class2idx = {'normal':0, 'fw':1, 'fm':2, 'dpi':3, 'ids':4, 'none':5}
idx2class = ['normal', 'fw', 'fm', 'dpi', 'ids', 'none']
dict = {'class2idx':class2idx, 'idx2class':idx2class}

with open(lad1_dict_path, 'wb') as fp:
    pkl.dump(dict, fp)
with open(lad2_dict_path, 'wb') as fp:
    pkl.dump(dict, fp)
