"""Protip: use 'verbose=True/False' to toggle printouts for CSVFolderDataset, earlyStopping, and 
ClassifierTrainer (the latter as of TorchEEG 1.1.3)."""

#Setup:

import os
from torcheeg.datasets import CSVFolderDataset
from torcheeg import transforms
from torcheeg.models import LGGNet
from torcheeg.trainers import ClassifierTrainer
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
from torcheeg.model_selection import * #NOTE: KFoldGroupbyTrial <== WATCH OUT FOR 'by'/'By'
import mne
import pytorch_lightning as pl #For tracking training epochs

eeg_type = "DAAMEEceegrid"
model_name = "LGGNET"
model_type = "(GNN)"

#Paths
dataset_csv_path = './dataConverted_ceegrid_norm_noFpz/dataset_'

def correctReadFn(file_path, **kwargs): #NOTE: IT FAILS WHEN ONLY ONE EPOCH PER FILE
    file_path = file_path.replace("\\", "/")
    raw = mne.io.read_raw(file_path)
    #Convert raw to epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=1.015) #ASSUMING TRIMMED TO 28S
    #Return EEG data
    return epochs

n_splits_KFolds = 10
#Define split strategies
splits = {"KFold": KFold(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldGroupbyTrial": KFoldGroupbyTrial(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldCrossTrial": KFoldCrossTrial(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldCrossSubject":KFoldCrossSubject(n_splits=n_splits_KFolds, shuffle=True),
          "LeaveOneSubjectOut":LeaveOneSubjectOut()}

#Region list- directly input to model, NOT used in transforms:
CEEGRID_REGION_LIST = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19]]
offline_transform=transforms.To2d()
online_transform=transforms.ToTensor()
maps = [{'HV': 1, 'LV': 0}, {'HA': 1, 'LA': 0}, {'HD': 1, 'LD': 0}]

#Setup for later creating results files:
basePath = os.getcwd()
def ensure_tsv_header(file_path, header_fields): #To ensure files exist and check contents
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("\t".join(header_fields) + "\n")

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
for label_idx, label_name in enumerate(["valence", "arousal", "dominance"]):
      if label_name != "valence": #In case you only want to focus on one.
          continue
      else:
            
            print(f"\nTraining for {label_name} classification:")
            
            dataset_csv_path_thisDim = dataset_csv_path + label_name[:3] + ".csv"
            io_path = f'./cache_{eeg_type}_{label_name}_{model_name}'
            
            label_transform=transforms.Compose([transforms.Select('label'),
                                                transforms.Mapping(maps[label_idx])])
                                       
            dataset = CSVFolderDataset(io_path=io_path,
                                       csv_path=dataset_csv_path_thisDim,    
                                       read_fn = correctReadFn,
                                       offline_transform=offline_transform,
                                       online_transform=online_transform,
                                       label_transform=label_transform,
                                       num_worker=6,
                                       io_mode="lmdb")
            
            dataset.info['subject_id'] = dataset.info['id'].str.extract(r'(sub-\d+)')
            dataset.info['trial_id'] = dataset.info.index // 28
            trial_counts = dataset.info.groupby('trial_id').size()
            
            #Ensure results files exist:
            resultsFile = os.path.join(basePath, f"{eeg_type}_Results_{model_name}_{label_name}.tsv")
            epochResultsFile = os.path.join(basePath, f"{eeg_type}_EpochResults_{model_name}_{label_name}.tsv")
            ensure_tsv_header(resultsFile, ["Label", "Split", "Accuracy (%)", "F1-score (%)"])
            ensure_tsv_header(epochResultsFile, ["Label", "Split", "Epochs (Mean)", "Epochs (STD)"])
                        
            #Loop over the splits
            for splitname, split in splits.items():
                print("Solving for ",label_name, " using split: ", splitname)
                accuracies = [] #Used for averaging
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
                            train_indices = train_test_dataset.info.index[train_test_dataset.info['clip_id']
                                                                          .isin(overlapping_clip_ids)].tolist()

                            #Create datasets
                            train_dataset = Subset(train_test_dataset, train_indices)
                            
                            #Create DataLoaders for each set
                            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
                            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)                                        
                        
                            #Define your model
                            model = LGGNet(region_list=CEEGRID_REGION_LIST,
                                           chunk_size=203,
                                           num_electrodes=20,
                                           hid_channels=32,
                                           num_classes=2)
                            
                            #Early stopping callback
                            early_stopping = EarlyStopping(min_delta=0.00,
                                                           monitor='val_accuracy',
                                                           patience=10, #5-10 recommended
                                                           mode='max')
                            
                            #Set up the trainer with metrics
                            trainer = ClassifierTrainer(model=model,
                                                        num_classes=2,
                                                        lr=1e-4,
                                                        weight_decay=1e-4,
                                                        metrics=['accuracy', 'f1score'],
                                                        accelerator="cpu")
                    
                            #Train the model with early stopping
                            trainer.fit(train_loader, val_loader, max_epochs=50,
                                        callbacks=[early_stopping])
                            
                            epoch_tracker = EpochTracker() #Define here so it starts at 0.
                            
                            #Train the model with early stopping
                            trainer.fit(train_loader, val_loader, max_epochs=50,
                                        callbacks=[early_stopping,epoch_tracker])
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