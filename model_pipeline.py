import numpy as np
import pandas as pd
import torch
import random
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load custom functions
from helpers import *
from train_functions import *

###########
## Setup ##
###########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == "cuda":
  print(torch.cuda.get_device_name())
if device.type == "cpu":
   print("Number of cpus:", os.cpu_count())

# For reproducibility 
np.random.seed(2019)
torch.manual_seed(2019)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2019)
    torch.backends.cudnn.deterministic = True # use deterministic algorithms to make it reproducible
    torch.backends.cudnn.benchmark = False

##################################
## Load and Preprocess the Data ##
##################################

df = pd.read_feather("./data/claims.feather")

# We reduce the dataset for demonstration purposes
ids = df.groupby('AY')[['AY', 'ClNr']].apply(lambda x: x.sample(n=min(len(x), 200), random_state=1)).reset_index(drop=True).ClNr.unique()
df = df[df.ClNr.isin(ids)]

# Prepare data
df = prep_data(df)

# Select only columns used in the model
meta_cols = ['ClNr', 'zero_amount', 'AY']
cat_vars = ['LoB', 'cc', 'inj_part'] 
num_vars_static = ['age', 'RepDel']
num_vars = ['dev_year_predictor']
time_vars = ['ay', 'dev_year']
target = ['cum_loss']

df_prep = df[meta_cols + target + cat_vars + num_vars_static + num_vars + time_vars]

# Get mapping list for the development period predictor
dev_vars = df_prep[['dev_year', 'dev_year_predictor']].drop_duplicates()

# Label Encoding for categorical features
df_prep, _ = perform_label_encoding(df_prep, columns_to_encode=cat_vars)
df_prep[cat_vars] = df_prep[cat_vars].astype("int")

# Create triangle 
triangle = square_to_triangle(df_prep, ay_col='ay', time_col='dev_year')

# Visualize aggregated data
df_prep.pivot_table(index='AY', columns='dev_year', values=target, aggfunc='sum', fill_value=np.nan)
triangle.pivot_table(index='AY', columns='dev_year', values=target, aggfunc='sum', fill_value=np.nan)

# Removal of contracts with only one timestamp
training_data = rm_single_claims(triangle, ay_col='ay')
training_data.pivot_table(index='ay', columns='dev_year', values=target, aggfunc='sum', fill_value=np.nan)

# Train and Validation Split
train_df, val_df = train_val_split_temporal(training_data, ay_col='ay', id_col='ClNr', train_size=0.8)
train_df.pivot_table(index='ay', columns='dev_year', values=target, aggfunc='sum', fill_value=np.nan)
val_df.pivot_table(index='ay', columns='dev_year', values=target, aggfunc='sum', fill_value=np.nan)

# Check distribution per ay
train_df.groupby(['ay'])['ClNr'].nunique()
val_df.groupby(['ay'])['ClNr'].nunique()

# Scaling the training data and using it to transform the validation set
num_scalers_train = get_standard_scalers(train_df, target + num_vars_static + num_vars)

train_df = apply_scaling(train_df, num_scalers_train)
val_df = apply_scaling(val_df, num_scalers_train)

# Now expand the time series to construct the model in a more readable way
train_df_expand = train_df.groupby("ClNr")[train_df.columns].apply(expand_time_series).reset_index(drop=True)
train_df_expand['ClNr_sub'] = train_df_expand['ClNr_sub'].factorize()[0].astype("int")

val_df_expand = val_df.groupby("ClNr")[val_df.columns].apply(expand_time_series).reset_index(drop=True)
val_df_expand['ClNr_sub'] = val_df_expand['ClNr_sub'].factorize()[0].astype("int")

# For parallel computing one can also use:
# train_df_expand = parallelize_expand_df(train_df, process_chunk_expand, n_cores=12, id='ClNr')
# val_df_expand = parallelize_expand_df(val_df, process_chunk_expand, n_cores=12, id='ClNr')
              
# Initialize grid for random search 
n_inputs = len(num_vars) + 1
n_nums_static = len(num_vars_static)
cat_sizes = df[cat_vars].nunique().tolist()
embed_dims = [np.ceil(x**0.5).astype('int') for x in cat_sizes]
n_cats = len(cat_vars)
n_nums = 1 + len(num_vars)

config = {
    "n_inputs": len(num_vars) + 1,
    "hidden_size_static": tune.choice([32, 64, 128]),
    "hidden_size_lstm": tune.choice([32, 64, 128]),
    "hidden_size_comb": tune.choice([32, 64, 128]),
    "n_static_nums": n_nums_static,
    "num_layers": tune.choice([1, 2]),
    "cat_sizes": cat_sizes,
    "embed_dims": embed_dims,
    "dropout": tune.choice([0.1, 0.3, 0.5]),
    "batch": 2048,
    "lr": tune.loguniform(1e-2, 1e-5),
    "weight_decay": 0.0001,
    "step_size": 10,
    "epochs": 30,
    "patience": 10,
    'alpha': tune.choice([0.1, 0.5, 1, 1.5])
}
 
ray.init(num_cpus=32, num_gpus=0, include_dashboard=True)

start_time = time.time()
analysis = tune.run(
    with_parameters(train_model, train_df=train_df_expand, val_df=val_df_expand),
    config=config,
    num_samples=32,
    resources_per_trial={"cpu": 1, "gpu": 0},
    metric="val_loss",
    mode="min",
    verbose=1
)
end_time = time.time() - start_time
print(f"Time: {end_time}")

best_config = analysis.get_best_config(metric="val_loss", mode="min")
print(best_config)
ray.shutdown()

# Use the best hyperparameters to train the model

best_hyper = {
    "n_inputs": len(num_vars) + 1,
    "hidden_size_static": 64,
    "hidden_size_lstm": 128,
    "hidden_size_comb": 32,
    "n_static_nums": n_nums_static,
    "num_layers": 1,
    "cat_sizes": cat_sizes,
    "embed_dims": embed_dims,
    "dropout": 0.1,
    "batch": 4096,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "step_size": 100,
    "epochs": 30,
    "patience": 15,
    'alpha': 0.5
}

net = train_model(best_hyper, train_df=train_df_expand, val_df=val_df_expand, tune=False, return_model=True, target='cum_loss', incr=False, zero_col='zero_amount', num_workers=1)    

## Use the model to predict the training and validation set ##

# Predict Training Data   
train_pred_df = predict(df=train_df_expand, net=net, target='cum_loss', id_col='ClNr_sub', batch=4096, num_workers=1)
train_pred_df[['pred_reg', 'pred_clf', 'last_val']] = train_pred_df[['pred_reg', 'pred_clf', 'last_val']].astype(float)
train_pred_df[['ClNr_sub', 'dev_year']] = train_pred_df[['ClNr_sub', 'dev_year']].astype(int)

# Merge predictions to training data 
train_pred_df = train_df_expand.copy().merge(train_pred_df, how='left', on=['ClNr_sub', 'dev_year'])

# Inverse Scaling
train_pred_df['cum_loss'] = num_scalers_train['cum_loss'].inverse_transform(train_pred_df[['cum_loss']])
train_pred_df['pred_reg'] = num_scalers_train['cum_loss'].inverse_transform(train_pred_df[['pred_reg']])
train_pred_df['last_val'] = num_scalers_train['cum_loss'].inverse_transform(train_pred_df[['last_val']])

# Filter data
train_preds = train_pred_df.query('pred_reg == pred_reg').copy()
predicted_probs = np.array(train_preds['pred_clf'])
true_binary_labels = np.array(train_preds['zero_amount'])

# Compute ROC curve and AUC for classification task
fpr, tpr, thresholds = roc_curve(true_binary_labels, predicted_probs)
roc_auc = roc_auc_score(true_binary_labels, predicted_probs)

# Determine optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f'AUC: {roc_auc:.2f}')
print(f'Optimal Threshold: {optimal_threshold:.2f}')

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='black', label='Optimal Threshold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

predicted_labels = np.where(predicted_probs > optimal_threshold, 1, 0)
cm = confusion_matrix(true_binary_labels, predicted_labels)
ConfusionMatrixDisplay(cm).plot()

# Calculate final prediction
train_pred_df['final_pred'] = np.where(train_pred_df['pred_clf'] > optimal_threshold, train_pred_df['pred_reg'], train_pred_df['last_val'])

# Calculate RMSE
calculate_rmse(train_pred_df.query("pred_reg == pred_reg"), true_col='cum_loss', pred_col='final_pred')
train_pred_df.query("pred_reg == pred_reg").groupby('dev_year').apply(calculate_rmse, true_col='cum_loss', pred_col='final_pred')

# Percentage error of aggregated portfolio
t = train_pred_df.query('pred_reg == pred_reg').pivot_table(index='ay', columns='dev_year', values='cum_loss', aggfunc='sum', fill_value=np.nan)
p = train_pred_df.query('pred_reg == pred_reg').pivot_table(index='ay', columns='dev_year', values='final_pred', aggfunc='sum', fill_value=np.nan)
(p-t)/t


# Predict validation data
val_pred_df = predict(df=val_df_expand, net=net, target='cum_loss', id_col='ClNr_sub', batch=4096, num_workers=1)
val_pred_df[['pred_reg', 'pred_clf', 'last_val']] = val_pred_df[['pred_reg', 'pred_clf', 'last_val']].astype(float)
val_pred_df[['ClNr_sub', 'dev_year']] = val_pred_df[['ClNr_sub', 'dev_year']].astype(int)

# Merge predictions to validation data 
val_pred_df = val_df_expand.copy().merge(val_pred_df, how='left', on=['ClNr_sub', 'dev_year'])

# Inverse Scaling
val_pred_df['cum_loss'] = num_scalers_train['cum_loss'].inverse_transform(val_pred_df[['cum_loss']])
val_pred_df['pred_reg'] = num_scalers_train['cum_loss'].inverse_transform(val_pred_df[['pred_reg']])
val_pred_df['last_val'] = num_scalers_train['cum_loss'].inverse_transform(val_pred_df[['last_val']])

# Calculate final prediction
val_pred_df['final_pred'] = np.where(val_pred_df['pred_clf'] > optimal_threshold, val_pred_df['pred_reg'], val_pred_df['last_val'])

# Calculate RMSE
calculate_rmse(val_pred_df.query("final_pred == final_pred"), true_col='cum_loss', pred_col='final_pred')
val_pred_df.query("final_pred == final_pred").groupby('dev_year').apply(calculate_rmse, true_col='cum_loss', pred_col='final_pred')

# Percentage error of aggregated portfolio
t = val_pred_df.query('final_pred == final_pred').pivot_table(index='ay', columns='dev_year', values='cum_loss', aggfunc='sum', fill_value=np.nan)
p = val_pred_df.query('final_pred == final_pred').pivot_table(index='ay', columns='dev_year', values='final_pred', aggfunc='sum', fill_value=np.nan)
(p-t)/t

## Fit model to train- and validation set ##

# Scaling the complete training data 
train_val = pd.concat((train_df, val_df))
num_scalers = get_standard_scalers(train_val, target + num_vars_static + num_vars)
final_train_df = apply_scaling(train_val, num_scalers)

# Now expand the time series to construct the model in a more readable way
final_train_df_expand = parallelize_expand_df(final_train_df, process_chunk_expand, n_cores=64, id='ClNr')
final_train_df_expand['ClNr_sub'] = final_train_df_expand['ClNr_sub'].factorize()[0].astype("int")

# Train the model based on trainin and validation set
final_net = fit_model(best_hyper, train_df=final_train_df_expand, num_workers=76)

# Fill up the test triangle
square = triangle[triangle.ClNr.isin(test.ClNr)].copy()
square.pivot_table(index='ay', columns='dev_year', values='cum_loss', aggfunc='sum', fill_value=np.nan)
test = df[df.ClNr.isin(test.ClNr)].copy()

max_ay = square['ay'].max()
min_ay = square['ay'].min()

# Apply scaling
square = apply_scaling(square, num_scalers)

dev_year_scaler = {key: num_scalers[key] for key in ['dev_year_predictor']}
dev_years_scaled = apply_scaling(dev_vars, dev_year_scaler)

square['pred_reg'] = np.nan
square['pred_clf'] = np.nan

for year in range(1,max_ay+1):
    for no_preds in range(1, year+1):

        print(year)
        print(no_preds)
        
        # fill up all sequences with ay = 1 -> need one prediciton
        subset = square[square['ay'] == year]
        

        preds = predict(df=subset, net=net, target='cum_loss', id_col='ClNr', batch=4096, num_workers=1, rm_last_val=False)

        preds[['pred_reg', 'pred_clf', 'last_val']] = preds[['pred_reg', 'pred_clf', 'last_val']].astype(float)
        preds[['ClNr', 'dev_year']] = preds[['ClNr', 'dev_year']].astype(int)
        preds['cum_loss'] = np.where(preds['pred_clf'] > optimal_threshold, preds['pred_reg'], preds['last_val'])
       
        preds.drop(columns=['last_val'], inplace=True)
        
        # Get original sequences and modify dynamic attributes
        max_time = subset['dev_year'].max() # max dev year
        max_dev_rows = subset[subset['dev_year'] == max_time].copy()
        max_dev_rows['dev_year'] += 1
        
        # Remove target column and macro vars
        max_dev_rows = max_dev_rows.drop(columns=['cum_loss', 'dev_year_predictor', 'pred_reg', 'pred_clf', 'dev_year'])

        # Add dynmaic variables 
        tmp = max_dev_rows.merge(preds, on='ClNr', how='left')
        tmp = tmp.merge(dev_years_scaled, on='dev_year', how='left')

        # Append data with predictions to apply recursive multi-step ahead forecasting
        square = pd.concat([square, tmp], axis=0)

# Inverse scaling
square['cum_loss'] = num_scalers['cum_loss'].inverse_transform(square[['cum_loss']])
square['pred_reg'] = num_scalers['cum_loss'].inverse_transform(square[['pred_reg']])

# Aggregated results
preds_agg = square.pivot_table(index='ay', columns='dev_year', values='cum_loss', aggfunc='sum', fill_value=np.nan)
true_agg = test.pivot_table(index='ay', columns='dev_year', values='cum_loss', aggfunc='sum', fill_value=np.nan)
(preds_agg - true_agg) / true_agg

t = test.query("dev_year == 11").cum_loss.sum() 
p = square.query("dev_year == 11").cum_loss.sum() 
p/t

# Save models
# torch.save(final_net, '/home/h067341/rsp-projects/ap.nn_loss_reserving/nn_loss_reserving/custom/models/full_model_f')
# torch.save(final_net.state_dict(), '/home/h067341/rsp-projects/ap.nn_loss_reserving/nn_loss_reserving/custom/models/full_model_f_dict.pth')

# torch.save(net, '/home/h067341/rsp-projects/ap.nn_loss_reserving/nn_loss_reserving/custom/models/train_model_f')
# torch.save(net.state_dict(), '/home/h067341/rsp-projects/ap.nn_loss_reserving/nn_loss_reserving/custom/models/train_model_f_dict.pth')
