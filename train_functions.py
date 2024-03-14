import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
import ray
from ray import train
from ray.tune import with_parameters
from ray import tune
import time

class ClaimsDataset(Dataset):
    # Receives a pandas dataframe and stores it for later use
    # Extracts a list of unique claim ids

    def __init__(self, dataframe, id_cols, features, target, time_col, incr, zero_col, rm_last_val=True):
        self.dataframe = dataframe
        self.id_cols = id_cols
        self.features = features
        self. time_col = time_col
        self.target = target
        self.incr = incr
        self.zero_col = zero_col
        self.ids = dataframe[id_cols].unique()
        self.rm_last_val = rm_last_val

    # Return the number of unique claims in the dataset
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        # Extract a specific claim
        contract_data = self.dataframe[self.dataframe[self.id_cols] == self.ids[idx]]

        # Sort time series
        contract_data = contract_data.sort_values(by = self.time_col)

        # Pull out id
        sequence_id = self.ids[idx]

        # Create torch tensor
        if self.rm_last_val:
            sequence = torch.tensor(contract_data[[self.target]+self.features].values[:-1], dtype = torch.float32)
            target = torch.tensor(contract_data[self.target].values[-1], dtype = torch.float32)
            
            if self.incr:
                change = contract_data[self.zero_col].values[-1] == 1
            else:
                change = contract_data[self.target].values[-1] != contract_data[self.target].values[-2]
        
        else:
            sequence = torch.tensor(contract_data[[self.target]+self.features].values, dtype = torch.float32)
            target = torch.tensor(0)
            change = torch.tensor([0.])
            
        clf_target = torch.tensor([change], dtype=torch.float32)


        return sequence_id, sequence, target, clf_target

def collate_fn(batch):
    # Separate sequences and targets from the batch
    sequence_ids, sequences, targets, clf_targets = zip(*batch)
    
    # Pad sequences to have the same length
    padded_sequences = pad_sequence(sequences, batch_first=True)
    
    # Convert targets into a tensor
    targets = torch.stack(targets)
    clf_targets = torch.stack(clf_targets)

    
    # Calculate sequence lengths for dynamic unpadding later (if necessary)
    lengths = torch.tensor([len(seq) for seq in sequences])

    return sequence_ids, padded_sequences, targets, clf_targets, lengths

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        
    def forward(self, lstm_output):
        # Calculate attention scores
        attention_scores = torch.bmm(lstm_output, lstm_output.transpose(1, 2))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # Multiply the lstm outputs by attention weights
        attended_lstm_output = torch.bmm(attention_weights, lstm_output)
        
        return attended_lstm_output, attention_weights
    
class MultiTaskLSTM(nn.Module):

    def __init__(self, input_size, 
                 hidden_size_static, hidden_size_lstm, hidden_size_comb,
                 output_size_claim, output_size_prob, 
                 n_static_nums, num_layers, cat_sizes, embed_dims, dropout=0.0):
        super(MultiTaskLSTM, self).__init__()

        assert len(cat_sizes) == len(embed_dims), "Length of cat_sizes and embed_dims should be the same."
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, embed_dim) for cat_size, embed_dim in zip(cat_sizes, embed_dims)])
        total_embed_dim = sum(embed_dims) + n_static_nums

        # Fully connected layer for static features
        self.fc_static = nn.Linear(total_embed_dim, hidden_size_static)
        self.dropout_static = nn.Dropout(dropout)

        self.attention = DotProductAttention()

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size_lstm, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer for concatenated output
        self.fc_concatenated = nn.Linear(hidden_size_static + hidden_size_lstm, hidden_size_comb)
        self.dropout_concatenated = nn.Dropout(dropout)

        # Output layers for two tasks
        self.fc_claim = nn.Linear(hidden_size_comb, output_size_claim)
        self.fc_prob = nn.Linear(hidden_size_comb, output_size_prob)
    
    def forward(self, x, n_nums, n_nums_static, n_cats, lengths):
        num_features = x[:, :, :n_nums]
        num_static_features = x[:, 0, n_nums:(n_nums+n_nums_static)]
        cat_features = x[:, 0, (n_nums+n_nums_static):].long()

        embedded_features = [embed(cat_features[:, i]) for i, embed in enumerate(self.embeddings)]
        embedded_features = torch.cat(embedded_features, dim=1)
        static_vars = torch.cat((num_static_features, embedded_features), dim=1)

        static_output = self.fc_static(static_vars)
        static_output = F.relu(static_output)
        static_output = self.dropout_static(static_output)

        packed = pack_padded_sequence(num_features, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out, att_weights = self.attention(lstm_out)
        
        batch_size, _, hidden_dim = lstm_out.shape
        batch_indices = torch.arange(0, batch_size)
        last_outputs = lstm_out[batch_indices, lengths-1, :]

        concatenated_output = torch.cat([static_output, last_outputs], dim=1)

        concatenated_output = self.fc_concatenated(concatenated_output)
        concatenated_output = F.relu(concatenated_output)
        concatenated_output = self.dropout_concatenated(concatenated_output)

        # Task-specific outputs
        output_claim = self.fc_claim(concatenated_output)
        output_prob = torch.sigmoid(self.fc_prob(concatenated_output))
        
        return output_claim, output_prob

def train_model(config, train_df=None, val_df=None, tune=True, return_model=False,
                target='cum_loss',  incr=False, zero_col='zero_amount', num_workers=2,
                num_vars =['dev_year_predictor'], num_vars_static=['age', 'RepDel'], cat_vars=['LoB', 'cc', 'inj_part'],
                n_nums=2, n_nums_static=2, n_cats=3,
                device="cpu"):
    
    # Create the training and validation datasets and loaders
    train_set = ClaimsDataset(train_df, id_cols = "ClNr_sub", features = num_vars + num_vars_static + cat_vars, target=target, time_col="dev_year", incr=incr, zero_col=zero_col)
    train_loader = DataLoader(train_set, batch_size=config['batch'], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    val_set = ClaimsDataset(val_df, id_cols = "ClNr_sub", features = num_vars + num_vars_static + cat_vars, target=target, time_col="dev_year", incr=incr, zero_col=zero_col)
    val_loader = DataLoader(val_set, batch_size=config['batch'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Initialize the weights for the two tasks 
    Weightloss1 = torch.tensor([1], requires_grad=True, dtype=torch.float, device=device)
    Weightloss2 = torch.tensor([1], requires_grad=True, dtype=torch.float, device=device)
    
    params = [Weightloss1, Weightloss2]

    # Initiate Model
    model = MultiTaskLSTM(input_size=config['n_inputs'],
                          hidden_size_static=config['hidden_size_static'],
                          hidden_size_lstm=config['hidden_size_lstm'],
                          hidden_size_comb=config['hidden_size_comb'],
                          output_size_claim=1,
                          output_size_prob=1,
                          n_static_nums=config['n_static_nums'],
                          num_layers=config['num_layers'],
                          cat_sizes=config['cat_sizes'],
                          embed_dims=config['embed_dims'],
                          dropout=config['dropout'])
    model.to(device)
    
    # Set up Optimizer
    opt_params = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    opt_weights = torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Loss functions
    criterion_reg = nn.MSELoss(reduction='none')
    criterion_clf = nn.BCELoss(reduction='none')
    grad_loss = nn.L1Loss()
    
    alpha = config['alpha']
 
    num_epochs = config['epochs']
    
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = config['patience']  # number of epochs to wait before stopping
    
    for epoch in range(num_epochs):
        
        print(epoch)
        
        start_time = time.time()
        
        model.train()
        total_train_loss = 0
        total_reg_loss = 0
        total_clf_loss = 0
        coef=0
        
        for id, input, target, clf_target, lengths in train_loader:
            
            input = input.to(device)
            target = target.to(device)
            clf_target = clf_target.to(device)
            
            # Forward pass
            reg_out, clf_out = model(input, n_nums, n_nums_static, n_cats, lengths)
            
            # Calculate the weighted losses for both tasks
            l1 = params[0] * criterion_reg(reg_out.squeeze(dim=1), target).mean()
            l2 = params[1] * criterion_clf(clf_out, clf_target).mean()
            loss = torch.div(torch.add(l1,l2), 2)
            
            
            # Initial losses at t=0
            if epoch == 0:
                l01 = l1.data
                l02 = l2.data
            
            opt_params.zero_grad()
            loss.backward(retain_graph=True)
            
            # Getting gradients of the first layers of each tower and calculate their l2-norm 
            param = list(model.parameters())
            G1R = torch.autograd.grad(l1, param[0], retain_graph=True, create_graph=True)
            G1 = torch.norm(G1R[0], 2)
            G2R = torch.autograd.grad(l2, param[0], retain_graph=True, create_graph=True)
            G2 = torch.norm(G2R[0], 2)
            G_avg = torch.div(torch.add(G1, G2), 2)
            
            # Calculating relative losses 
            lhat1 = torch.div(l1,l01)
            lhat2 = torch.div(l2,l02)
            lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)
            
            # Calculating relative inverse training rates for tasks 
            inv_rate1 = torch.div(lhat1,lhat_avg)
            inv_rate2 = torch.div(lhat2,lhat_avg)
        
            # Calculating the constant target for Eq. 2 in the GradNorm paper
            C1 = G_avg*(inv_rate1)**alpha
            C2 = G_avg*(inv_rate2)**alpha
            C1 = C1.squeeze().detach()
            C2 = C2.squeeze().detach()
            
            opt_weights.zero_grad()
  
            Lgrad = torch.add(grad_loss(G1, C1),grad_loss(G2, C2))
            Lgrad.backward()
            
            # Updating loss weights
            opt_weights.step()
            
            # Clamping the task weights to be non-negative
            Weightloss1.data = torch.clamp(Weightloss1.data, min=0.1)
            Weightloss2.data = torch.clamp(Weightloss2.data, min=0.1)
            
            # Updating model weights
            opt_params.step()
            
            # Renormalizing the losses weights
            coef = 2 / (Weightloss1 + Weightloss2)
            Weightloss1.data = coef * Weightloss1.data
            Weightloss2.data = coef * Weightloss2.data

            
            total_train_loss += loss.item()
            total_reg_loss += l1.item()
            total_clf_loss += l2.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        avg_clf_loss = total_clf_loss / len(train_loader)
        print(f"Weight1: {Weightloss1}")
        print(f"Weight2: {Weightloss2}")

        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        
        total_val_loss = 0
        total_val_reg_loss = 0
        total_val_clf_loss = 0
        
        with torch.inference_mode():
            
            for id, input, target, clf_target, lengths in val_loader:
                               
                input = input.to(device)
                target = target.to(device)
                clf_target = clf_target.to(device)
            
                # Forward pass
                reg_out, clf_out = model(input, n_nums, n_nums_static, n_cats, lengths)
                
                # Calculate the losses for both tasks
                l1_val = params[0] * criterion_reg(reg_out.squeeze(dim=1), target).mean()
                l2_val = params[1] * criterion_clf(clf_out, clf_target).mean()
                
                loss_val = torch.div(torch.add(l1_val,l2_val), 2)
                    
                total_val_loss += loss_val.item()
                total_val_reg_loss += l1_val.item()
                total_val_clf_loss += l2_val.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_reg_loss = total_val_reg_loss / len(val_loader)
        avg_val_clf_loss = total_val_clf_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epochs_without_improvement >= patience:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Reg Loss: {avg_val_reg_loss}, Clf Loss: {avg_val_clf_loss}')

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}. No improvement in validation loss for {patience} consecutive epochs.")
            break
          
        if epoch % 10 == 0:
            end_time = time.time() - start_time
            print(f"Time for 1 epochs: {end_time}")
        
        if tune:
            
            train.report({'val_loss': avg_val_loss, "train_loss": avg_train_loss, 
                        "avg_val_reg_loss": avg_val_reg_loss,
                        "avg_val_clf_loss": avg_val_clf_loss,
                        "avg_train_reg_loss": avg_reg_loss,
                        "avg_train_clf_loss": avg_clf_loss,
                        "Weightloss1": Weightloss1.item(),
                        "Weightloss2": Weightloss2.item()})
            
    if return_model:
        return model

def fit_model(config, train_df=None, target='cum_loss', incr=False, zero_col='zero_amount', num_workers=2,
              num_vars =['dev_year_predictor'], num_vars_static=['age', 'RepDel'], cat_vars=['LoB', 'cc', 'inj_part'],
              n_nums=2, n_nums_static=2, n_cats=3,
              device="cpu"):
    
    # Create the training and validation datasets and loaders
    train_set = ClaimsDataset(train_df, id_cols = "ClNr_sub", features = num_vars + num_vars_static + cat_vars, target=target, time_col="dev_year", incr=incr, zero_col=zero_col)
    train_loader = DataLoader(train_set, batch_size=config['batch'], shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # Initialize the weights for the two tasks 
    Weightloss1 = torch.tensor([1], requires_grad=True, dtype=torch.float, device=device)
    Weightloss2 = torch.tensor([1], requires_grad=True, dtype=torch.float, device=device)
    
    params = [Weightloss1, Weightloss2]

    # Initiate Model
    model = MultiTaskLSTM(input_size=config['n_inputs'],
                          hidden_size_static=config['hidden_size_static'],
                          hidden_size_lstm=config['hidden_size_lstm'],
                          hidden_size_comb=config['hidden_size_comb'],
                          output_size_claim=1,
                          output_size_prob=1,
                          n_static_nums=config['n_static_nums'],
                          num_layers=config['num_layers'],
                          cat_sizes=config['cat_sizes'],
                          embed_dims=config['embed_dims'],
                          dropout=config['dropout'])
    model.to(device)
    

    # Set up Optimizer
    opt_params = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    opt_weights = torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Loss functions
    criterion_reg = nn.MSELoss(reduction='none')
    criterion_clf = nn.BCELoss(reduction='none')
    grad_loss = nn.L1Loss()
    
    alpha = config['alpha']
 
    num_epochs = config['epochs']
    
    train_losses = []

    patience = config['patience']  # number of epochs to wait before stopping
    
    for epoch in range(num_epochs):
        
        print(epoch)
        
        start_time = time.time()
        
        model.train()
        total_train_loss = 0
        total_reg_loss = 0
        total_clf_loss = 0
        coef=0
        
        for id, input, target, clf_target, lengths in train_loader:
            
            input = input.to(device)
            target = target.to(device)
            clf_target = clf_target.to(device)
            
            # Forward pass
            reg_out, clf_out = model(input, n_nums, n_nums_static, n_cats, lengths)
            
           
            # Calculate the weighted losses for both tasks
            l1 = params[0] * criterion_reg(reg_out.squeeze(dim=1), target).mean()
            l2 = params[1] * criterion_clf(clf_out, clf_target).mean()
            loss = torch.div(torch.add(l1,l2), 2)
            
            # Initial losses at t=0
            if epoch == 0:
                l01 = l1.data
                l02 = l2.data
            
            opt_params.zero_grad()
            loss.backward(retain_graph=True)
            
            # Getting gradients of the first layers of each tower and calculate their l2-norm 
            param = list(model.parameters())
            G1R = torch.autograd.grad(l1, param[0], retain_graph=True, create_graph=True)
            G1 = torch.norm(G1R[0], 2)
            G2R = torch.autograd.grad(l2, param[0], retain_graph=True, create_graph=True)
            G2 = torch.norm(G2R[0], 2)
            G_avg = torch.div(torch.add(G1, G2), 2)
            
            # Calculating relative losses 
            lhat1 = torch.div(l1,l01)
            lhat2 = torch.div(l2,l02)
            lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)
            
            # Calculating relative inverse training rates for tasks 
            inv_rate1 = torch.div(lhat1,lhat_avg)
            inv_rate2 = torch.div(lhat2,lhat_avg)
        
            # Calculating the constant target for Eq. 2 in the GradNorm paper
            C1 = G_avg*(inv_rate1)**alpha
            C2 = G_avg*(inv_rate2)**alpha
            C1 = C1.squeeze().detach()
            C2 = C2.squeeze().detach()
            
            opt_weights.zero_grad()

            Lgrad = torch.add(grad_loss(G1, C1),grad_loss(G2, C2))
            Lgrad.backward()
            
            # Updating loss weights
            opt_weights.step()
            
            # Clamping the task weights to be non-negative
            Weightloss1.data = torch.clamp(Weightloss1.data, min=0.1)
            Weightloss2.data = torch.clamp(Weightloss2.data, min=0.1)
            
            # Updating model weights
            opt_params.step()
            
            # Renormalizing the losses weights
            coef = 2 / (Weightloss1 + Weightloss2)
            Weightloss1.data = coef * Weightloss1.data
            Weightloss2.data = coef * Weightloss2.data
            
            total_train_loss += loss.item()
            total_reg_loss += l1.item()
            total_clf_loss += l2.item()

            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_reg_loss = total_reg_loss / len(train_loader)
        avg_clf_loss = total_clf_loss / len(train_loader)

        train_losses.append(avg_train_loss)
        print(f"Average Train Loss: {avg_train_loss}")
        print(f"Average Reg Loss: {avg_reg_loss}")
        print(f"Average Clf Loss: {avg_clf_loss}")
        print(f"Weight1: {Weightloss1}")
        print(f"Weight2: {Weightloss2}")
        
        if epoch % 10 == 0:
            end_time = time.time() - start_time
            print(f"Time for 1 epochs: {end_time}")
        
    return model

def predict(df, net, id_col='ClNr_sub', target='cum_loss', 
            num_vars=['dev_year_predictor'],
            num_vars_static=['age', 'RepDel'],
            cat_vars=['LoB', 'cc', 'inj_part'],
            batch = 4,
            num_workers=1,
            rm_last_val=True):
    
    train_set = ClaimsDataset(df, id_cols = id_col, features = num_vars + num_vars_static + cat_vars, target=target, time_col="dev_year", incr=False, zero_col='zero_amount', rm_last_val=rm_last_val)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    
    result = pd.DataFrame()
    for id, input, target, clf_target, lengths in train_loader:
                
            input = input.to(device)
            target = target.to(device)
            clf_target = clf_target.to(device)
                
            # Forward pass
            net.to(device)
            net.eval()
            with torch.inference_mode():
                reg_out, clf_out = net(input, n_nums, n_nums_static, n_cats, lengths)
            
            reg = reg_out.squeeze().cpu().detach().numpy()
            clf = clf_out.squeeze().cpu().detach().numpy()
            input = input.cpu().detach()
            
            block_indices = torch.arange(input.size(0))
            last_val = input[block_indices, lengths-1][:, 0].numpy()
            dev_year = lengths.numpy()
            np.array(id)
            
            res = pd.DataFrame({'pred_reg': reg, 'pred_clf': clf, 'last_val': last_val, 'dev_year': dev_year, id_col: np.array(id)})
            
            result = pd.concat((result, res))
            
    return result
    
