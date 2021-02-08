import ray
from ray import tune

data='cnsm_exp2_2'
encoder='rnn'

dir = data + '.' + encoder 

analysis = tune.Analysis(".ray_result/"+dir, default_metric = 'val_f1', default_mode = 'max')
best_logdir = analysis.get_best_logdir()
print("best trial log path:\n", best_logdir)
best_trial = analysis.trial_dataframes[best_logdir]

best_config = analysis.get_all_configs()[best_logdir]
print("best trial config:\n", best_config)

peak_epoch = best_trial['val_f1'].argmax()

look_epoch = int(5 * (peak_epoch / 5)+1) 

peak_test_f1 = best_trial['test_f1'][look_epoch]

print("best_f1: {} at {} where the best val_f1 shown".format(peak_test_f1, look_epoch))
