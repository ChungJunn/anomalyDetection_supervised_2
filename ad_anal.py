import ray
from ray import tune

analysis = tune.Analysis(".ray_result/cnsm_exp1.none", default_metric = 'val_f1', default_mode = 'max')
best_logdir = analysis.get_best_logdir()
print("best trial log path:\n", best_logdir)
best_trial = analysis.trial_dataframes[best_logdir]

best_config = analysis.get_all_configs()[best_logdir]
print("best trial config:\n", best_config)

peak_epoch = best_trial['val_f1'].argmax()
peak_accuracy = best_trial['val_f1'][peak_epoch]

print("best tiral peak result: {} at {}".format(peak_accuracy, peak_epoch))
