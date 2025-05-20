"""Protip: use 'verbose=True/False' to toggle printouts for DEAPDataset, earlyStopping, and 
ClassifierTrainer (the latter as of TorchEEG 1.1.3)."""

#Setup:

import os
import pandas as pd
from torcheeg.datasets import CSVFolderDataset
from torcheeg import transforms
from torcheeg.models import DGCNN
from torcheeg.trainers import ClassifierTrainer
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torcheeg.datasets.constants.utils import format_channel_location_dict
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
from torcheeg.model_selection import * #NOTE: KFoldGroupbyTrial <== WATCH OUT FOR 'by'/'By'
import mne
import pytorch_lightning as pl #For tracking training epochs

eeg_type = "ceegrid"
model_name = "DGCNN"
model_type = "(GNN)"

#Paths
dataset_csv_path = './dataConverted_ceegrid_norm_noFpz/dataset_'#"./dataConverted_normNotchFilt/dataset_"

def correctReadFn(file_path, **kwargs): #NOTE: IT FAILS WHEN ONLY ONE EPOCH PER FILE
    file_path = file_path.replace("\\", "/")
    raw = mne.io.read_raw(file_path)
    #Convert raw to epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=1.015) #ASSUMING TRIMMED TO 28.42S
    #Return EEG data
    return epochs

maps = [{'HV': 1, 'LV': 0}, {'HA': 1, 'LA': 0}, {'HD': 1, 'LD': 0},
        {'LVLALD':0,'LVLAHD':1,'LVHALD':2,'LVHAHD':3,'HVLALD':4,'HVLAHD':5,'HVHALD':6,'HVHAHD':7,}]

n_splits_KFolds = 10
#Define split strategies
splits = {"KFold": KFold(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldGroupbyTrial": KFoldGroupbyTrial(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldCrossTrial": KFoldCrossTrial(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldCrossSubject":KFoldCrossSubject(n_splits=n_splits_KFolds, shuffle=True),
          "LeaveOneSubjectOut":LeaveOneSubjectOut()}

basePath = os.getcwd()
#Ensure files exist, check contents:
resultsFile = os.path.join(basePath, f"{eeg_type}_Results_{model_name}_VAD.tsv")
epochResultsFile = os.path.join(basePath, f"{eeg_type}_EpochResults_{model_name}_VAD.tsv")

# Setup for TSV output
def ensure_tsv_header(file_path, header_fields):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("\t".join(header_fields) + "\n")

ensure_tsv_header(resultsFile, ["Label", "Split", "Accuracy (%)", "F1-score (%)"])
ensure_tsv_header(epochResultsFile, ["Label", "Split", "Epochs (Mean)", "Epochs (STD)"])

offline_transform=transforms.BandDifferentialEntropy(sampling_rate=200,
                                                     band_dict ={"delta": (1, 4),"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),"gamma": (30, 45)})
online_transform=transforms.ToTensor()
                

class EpochTracker(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epochs = 0  

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs += 1

##############################################################################################################################################################
##############################################################################################################################################################
#Deep learning loop:

#Define model and training for each label
for label_idx, label_name in enumerate(["valence", "arousal", "dominance","VAD"]):
   #   if label_name != "valence": #In case you only want to focus on one.
    #      continue
     # else:
            
            print(f"\nTraining for {label_name} classification:")
            io_path = f'./cache_{eeg_type}_{label_name}_{model_name}'
            
            if label_name == "VAD":
                num_classes = 8
            else:
                num_classes = 2
                        
            dataset_csv_path_thisDim = dataset_csv_path + label_name[:3] + ".csv"
            
            #Load dataset
            dataset = CSVFolderDataset(csv_path=dataset_csv_path_thisDim,    
                                       read_fn = correctReadFn,
                                       offline_transform=offline_transform,
                                       online_transform=online_transform,
                                       label_transform=transforms.Compose([
                                       transforms.Select('label'),
                                       transforms.Mapping(maps[label_idx])]),
                                       num_worker=7,
                                       io_mode="lmdb",
                                       io_path = io_path)
                    
            #For splitting by or across subject/trial:
            dataset.info['subject_id'] = dataset.info['id'].str.extract(r'(sub-\d+)')
            dataset.info['trial_id'] = dataset.info.index // 28
      #    trial_counts = dataset.info.groupby('trial_id').size()
            
            #Loop over the splits
            for splitname, split in splits.items():
                print("Solving for ",label_name, " using split: ", splitname)
                
                #Lists to store results/training data across folds:
                accuracies = []
                f1scores = []
                epochs_per_fold = []
                
                if isinstance(split, LeaveOneSubjectOut):
                    n_splits = len(dataset.info['subject_id'].unique())  #Num of subjs
                else:
                    n_splits = n_splits_KFolds  #Default value for other strategies
        
                for i, (train_test_dataset, val_dataset) in enumerate(split.split(dataset)):
                    for j, (train_val_dataset, test_dataset) in enumerate(split.split(dataset)):
                        if j == i+1 or (i == n_splits - 1 and j == 0):  #Rotating such that when Fold 0 is used for validation, fold 1 is used for test, ... fold 4 used for val, fold 0 used for test
                                                                     
                            #Map `clip_id` splits to dataset indices
                            train_test_clip_ids = train_test_dataset.info['clip_id'].unique()
                            train_val_clip_ids = train_val_dataset.info['clip_id'].unique()
                    
                            #Find overlapping clip_ids between train_val and train_test
                            overlapping_clip_ids = set(train_val_clip_ids).intersection(train_test_clip_ids)
                            
                            #Filter train_test indices for overlapping clip_ids
                            train_indices = [idx for idx, row in train_test_dataset.info.iterrows()
                                             if row['clip_id'] in overlapping_clip_ids]

                            #Create datasets
                            train_dataset = Subset(train_test_dataset, train_indices)
                            
                            #Create DataLoaders for each set
                            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
                            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)                  
                        
                            #Define your model
                            model = DGCNN(in_channels=5, num_electrodes=20, hid_channels=32, num_layers=2, num_classes=num_classes)
                            
                            #Early stopping callback
                            early_stopping = EarlyStopping(min_delta=0.00,
                                                           monitor='val_accuracy',
                                                           patience=10, #5-10 recommended
                                                           mode='max')
                            
                            #Set up the trainer with metrics
                            trainer = ClassifierTrainer(model=model,
                                                        num_classes=num_classes,
                                                        lr=1e-4,
                                                        weight_decay=1e-4,
                                                        metrics=['accuracy', 'f1score'],
                                                        accelerator="cpu")
                            
                            epoch_tracker = EpochTracker() #Define here so it starts at 0.
                            
                            #Train the model with early stopping
                            trainer.fit(train_loader, val_loader, max_epochs=50, callbacks=[early_stopping,epoch_tracker])
                            epochs_per_fold.append(epoch_tracker.epochs)

                            #Test the model
                            score = trainer.test(test_loader)[0]
                            accuracies.append(score["test_accuracy"])
                            f1scores.append(score["test_f1score"])
                                                            
                        else:
                            continue
                        
                #Calculate mean accuracy and F1-score
                mean_accuracy = np.mean(accuracies)*100
                mean_f1score = np.mean(f1scores)*100
                mean_epochs = np.mean(epochs_per_fold)
                std_epochs = np.std(epochs_per_fold)
                        
                #Write to file:
                with open(resultsFile, 'a') as f:
                    f.write(f"{label_name}\t{splitname}\t{mean_accuracy:.4f}\t{mean_f1score:.4f}\n")
        
                with open(epochResultsFile, 'a') as f:
                    f.write(f"{label_name}\t{splitname}\t{mean_epochs:.2f}\t{std_epochs:.2f}\n")