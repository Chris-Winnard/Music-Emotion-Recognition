"""Protip: use 'verbose=True/False' to toggle printouts for DEAPDataset, earlyStopping, and 
ClassifierTrainer (the latter as of TorchEEG 1.1.3)."""

#Setup:

import os
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms
from torcheeg.models import LSTM
from torcheeg.trainers import ClassifierTrainer
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
from torcheeg.model_selection import * #NOTE: KFoldGroupbyTrial <== WATCH OUT FOR 'by'/'By'
import pytorch_lightning as pl #For tracking training epochs

eeg_type = "DEAP"
model_name = "LSTM"
model_type = "(RNN)"

#Define split strategies
n_splits_KFolds = 10
splits = {"KFold": KFold(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldGroupbyTrial": KFoldGroupbyTrial(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldCrossTrial": KFoldCrossTrial(n_splits=n_splits_KFolds, shuffle=True),
          "KFoldCrossSubject":KFoldCrossSubject(n_splits=n_splits_KFolds, shuffle=True),
          "LeaveOneSubjectOut":LeaveOneSubjectOut()}

#For dataset initialization:
dataset_path = './data_preprocessed_python'
online_transform=transforms.ToTensor()

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
   #   if label_name != "valence": #In case you only want to focus on one.
    #      continue
     # else:
            
            print(f"\nTraining for {label_name} classification:")
            io_path = f'./cache_{eeg_type}_{label_name}_{model_name}'
                       
            dataset = DEAPDataset(io_path=io_path,
                      root_path='./data_preprocessed_python',
                      online_transform=online_transform,
                      label_transform=label_transform,
                      num_worker=6)
            
            #Ensure results files exist:
            resultsFile = os.path.join(basePath, f"{eeg_type}_Results_{model_name}_{label_name}.tsv")
            epochResultsFile = os.path.join(basePath, f"{eeg_type}_EpochResults_{model_name}_{label_name}.tsv")
            ensure_tsv_header(resultsFile, ["Label", "Split", "Accuracy (%)", "F1-score (%)"])
            ensure_tsv_header(epochResultsFile, ["Label", "Split", "Epochs (Mean)", "Epochs (STD)"])
            
            #Loop over the splits
            for splitname, split in splits.items():
                print("Solving for ",label_name, " using split: ", splitname)
                
                #Lists to store results/training data across folds:
                accuracies = []
                f1scores = []
                epochs_per_fold = []
                
                if isinstance(split, LeaveOneSubjectOut):
                    n_splits = 32#len(dataset.info['subject_id'].unique())  #Num of subjs
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
                            model = LSTM(num_electrodes=32, hid_channels=64, num_classes=2)
                            
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