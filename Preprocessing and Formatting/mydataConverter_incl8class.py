from pathlib import Path
import pandas as pd
import numpy as np
import mne
import mne_bids
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.stats import zscore
import asrpy
from badsMarker import *
import os
import random

class Parameters:
    def __init__(self):
        self.bidsRoot = './bids_dataset'
        self.task = 'emotion' 
        self.eeg_type = 'scalp' 
        self.output_dir = f'./dataConverted_{self.eeg_type}'
        if self.eeg_type == 'scalp':
            self.allSub = [f'{i:02d}' for i in range(1, 33) if i != 4]
            self.removeFpz = False
        elif self.eeg_type == 'ceegrid':
            self.allSub = [f'{i:02d}' for i in range(1, 33) if i in [4,16,21]]   
            self.removeFpz = True
        self.ds = 200
        self.trial_trim_start = 0.79
        self.trial_trim_end = 0.79
        self.eegFs = 1000
        self.eegPeriodAfterTrim = 30 - self.trial_trim_start - self.trial_trim_end
        self.normalise = True
        self.notchFilt = False
        self.interpBads = False
        self.doASR = False
        self.doICA = False 
        self.stricterICA = False
        self.labelling = 'absolute'#'absolute', 'relative', 'relativeCrossParticipant'
        self.ratingPerm = False

para = Parameters()
csv_rows_val = []
csv_rows_aro = []
csv_rows_dom = []
csv_rows_VAD = []

#For relative across participant labelling:
def collect_all_ratings(participant_list=para.allSub, bidsRoot=para.bidsRoot):
    all_ratings = {'valence': [], 'arousal': [], 'dominance': []}
    
    for subject in participant_list:
        subjFolder = f'sub-{subject}'
        labelsPath = os.path.join(bidsRoot, subjFolder, 'beh', f'{subjFolder}_task-{para.task}_beh.tsv')
        try:
            df = pd.read_csv(labelsPath, sep='\t')
            df = df[['valenceResp.response', 'arousalResp.response', 'dominanceResp.response']].dropna()
            all_ratings['valence'].extend(df['valenceResp.response'].values)
            all_ratings['arousal'].extend(df['arousalResp.response'].values)
            all_ratings['dominance'].extend(df['dominanceResp.response'].values)
        except FileNotFoundError:
            print(f"Behavioral file missing for {subject}")
    
    return all_ratings

def collect_all_ratings_with_permutation(participant_list=para.allSub, bidsRoot=para.bidsRoot):
    all_ratings = {'valence': [], 'arousal': [], 'dominance': []}
    
    for subject in participant_list:
        subjFolder = f'sub-{subject}'
        labelsPath = os.path.join(bidsRoot, subjFolder, 'beh', f'{subjFolder}_task-{para.task}_beh.tsv')
        try:
            df = pd.read_csv(labelsPath, sep='\t')
            df = df[['valenceResp.response', 'arousalResp.response', 'dominanceResp.response']].dropna()
            all_ratings['valence'].extend(df['valenceResp.response'].values)
            all_ratings['arousal'].extend(df['arousalResp.response'].values)
            all_ratings['dominance'].extend(df['dominanceResp.response'].values)
        except FileNotFoundError:
            print(f"Behavioral file missing for {subject}")
    
    #Permute ratings
    all_ratings['valence'] = random.sample(all_ratings['valence'], len(all_ratings['valence']))
    all_ratings['arousal'] = random.sample(all_ratings['arousal'], len(all_ratings['arousal']))
    all_ratings['dominance'] = random.sample(all_ratings['dominance'], len(all_ratings['dominance']))

    return all_ratings

#Step 2: Calculate global medians
def calculate_global_medians(all_ratings):
    medians = {'valence': np.median(all_ratings['valence']),
               'arousal': np.median(all_ratings['arousal']),
               'dominance': np.median(all_ratings['dominance'])}
    return medians

if para.ratingPerm == True:
    all_ratings = collect_all_ratings_with_permutation()
else:
    all_ratings = collect_all_ratings()
globalRatingMedians = calculate_global_medians(all_ratings)

for subject in para.allSub:
    subjFolder = f'sub-{subject}'
    bids_path = mne_bids.BIDSPath(root=para.bidsRoot, subject=subject, datatype='eeg',
                                  task=para.task, acquisition=para.eeg_type)
    eeg_raw = mne_bids.read_raw_bids(bids_path)
    eeg_raw.load_data()

    if para.removeFpz == True: #May choose to remove it from cEEGrid files.
        eeg_raw.drop_channels(['Fpz'])
        #Removed a channel, so given that we're using average ref it is best to reref:
        eeg_raw.set_eeg_reference(ref_channels="average")
        print("Fpz channel removed.")
        
    #Preprocessing
    eeg_filtered = eeg_raw.filter(l_freq=1, h_freq=45)
      
    if para.notchFilt == True:
        eeg_filtered = eeg_filtered.notch_filter((50),fir_window='blackman')

    if para.interpBads == True:
        subjFolder = "sub-" + subject
        eeg_filtered = badsMarker(eeg_filtered, subjFolder, para.eeg_type, para.task)
        eeg_filtered.set_eeg_reference(ref_channels="average") #Reref- labelled 'bads' not included in avg
        eeg_filtered = eeg_filtered.interpolate_bads()
        eeg_filtered.set_eeg_reference(ref_channels="average") #Reref again
    
    if para.doASR == True:
        asr = asrpy.ASR(sfreq=eeg_filtered.info["sfreq"], cutoff=20)
        asr.fit(eeg_filtered)
        eeg_preprocessed = asr.transform(eeg_filtered)
    
    if para.doICA == True: #Set up for if ASR and ICA both = True?
        ninterped = len(eeg_filtered.info['bads'])
        nchans = eeg_filtered.info['nchan']
        dims = nchans - ninterped - 1
        ica = ICA(n_components=dims, max_iter="auto", random_state=97,
                  method='infomax', fit_params=dict(extended=True))
        ica.fit(eeg_filtered)
    
        #ICLabel to estimate probabilities
        ic_labels = label_components(eeg_filtered, ica, method="iclabel")
    
        #Threshold for exclusion
        threshold = 0.5
        
        #List to store indices of components to exclude
        exclusion_indices = []
    
        if para.stricterICA == False: #Only remove ones labelled as eye artifacts
        
        #Iterate through each prediction and label
            for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                #Check if label is not 'brain' or 'other' and probability is over threshold
                if label in ['eye blink']:
                    #Mark component for exclusion
                    exclusion_indices.append(i)
            
        elif para.stricterICA == True:
        #Iterate through each prediction and label
            for i, (label, proba) in enumerate(zip(ic_labels['labels'], ic_labels['y_pred_proba'])):
                #Check if label is not 'brain' or 'other' and probability is over threshold
                if label not in ['brain'] or proba < threshold:
                    #Mark component for exclusion
                    exclusion_indices.append(i)
    
        #print("Components to exclude:", exclusion_indices)
    
        eeg_preprocessed = eeg_filtered.copy()
        ica.apply(eeg_preprocessed, exclude=exclusion_indices, n_pca_components=dims)
    
    if para.doASR != True and para.doICA != True:
        eeg_preprocessed = eeg_filtered

    #Events and epoching
    pathToEventsFile = Path(para.bidsRoot) / subjFolder / 'eeg' / f'{subjFolder}_task-{para.task}_acq-{para.eeg_type}_events.tsv'
    eventsData = pd.read_csv(pathToEventsFile, sep='\t')
    eventsArray = np.column_stack([(eventsData['onset'] * para.eegFs).astype(int),
                                   np.zeros(len(eventsData), dtype=int), eventsData['value']])
    tmin = para.trial_trim_start
    tmax = tmin + para.eegPeriodAfterTrim
    
    #Ignore part start/ends
    unique_events_raw = np.unique(eventsArray[:, 2])  #All event IDs in this particular task
    possibleTrialStartIDs = np.arange(1, 72, 2)  #ALL possible trial start IDs for all tasks
    missing_events = set(possibleTrialStartIDs) - set(unique_events_raw)  #Possible start IDs NOT in this particular task
    
    trialStartIDs = []
    for event in eventsArray:
        event_id = event[2]
        if event_id in possibleTrialStartIDs and event_id not in missing_events and event_id not in trialStartIDs:
            trialStartIDs.append(event_id)
    
    event_id = dict(zip(map(str, trialStartIDs), trialStartIDs))

    epochs = mne.Epochs(eeg_preprocessed, eventsArray, event_id=event_id, tmin=tmin,
                        tmax=tmax, baseline=None, preload=True,reject=None,detrend=1)
    
    #Behavioral Labels
    labelsPath = Path(para.bidsRoot) / subjFolder / 'beh' / f'{subjFolder}_task-{para.task}_beh.tsv'
    df = pd.read_csv(labelsPath, sep='\t').dropna(subset=["valenceResp.response"])
    
    if para.labelling == 'absolute':
        #Assign high/low labels
        valence_labels = (df['valenceResp.response'] > 5).astype(int)
        arousal_labels = (df['arousalResp.response'] > 5).astype(int)
        dominance_labels = (df['dominanceResp.response'] > 5).astype(int)
        
    elif para.labelling == 'relative':
      #Get the indices for the top 50% and bottom 50% for each column
      top_bottom_indices = {}
        
      for column in df.columns:
          #Sort the DataFrame based on the values in the current column
          sorted_df = df.sort_values(by=column)
            
          #Calculate the index to get the top 50% and bottom 50% of values
          top_50_percent_index = int(0.5 * len(sorted_df))
            
          #Get the indices for the top 50% and bottom 50%
          top_indices = sorted_df.index[-top_50_percent_index:]
          bottom_indices = sorted_df.index[:top_50_percent_index]
        
          #Store the indices in the dictionary
          top_bottom_indices[column] = {'top': top_indices, 'bottom': bottom_indices}
        
      valence_labels = [] #Go through the trials filling each list in
      arousal_labels = []
      dominance_labels = []
          
      for i in range(0, 16): #Includes prac trial
          if i in top_bottom_indices['valenceResp.response']['bottom']:
              valClass = 0
          elif i in top_bottom_indices['valenceResp.response']['top']:
              valClass = 1
            
          if i in top_bottom_indices['arousalResp.response']['bottom']:
              aroClass = 0
          elif i in top_bottom_indices['arousalResp.response']['top']:
              aroClass = 1
        
          if i in top_bottom_indices['dominanceResp.response']['bottom']:
              domClass = 0
          elif i in top_bottom_indices['dominanceResp.response']['top']:
              domClass = 1
        
          valence_labels.append(valClass)
          arousal_labels.append(aroClass)
          dominance_labels.append(domClass)

    elif para.labelling == 'relativeCrossParticipant':
        valence_labels = (df['valenceResp.response'] > globalRatingMedians['valence']).astype(int)
        arousal_labels = (df['arousalResp.response'] > globalRatingMedians['arousal']).astype(int)
        dominance_labels = (df['dominanceResp.response'] > globalRatingMedians['dominance']).astype(int)
        
    #Combine into eight-class labels (0 to 7)
    eight_class_labels = (valence_labels * 4) + (arousal_labels * 2) + dominance_labels
    
    label_map = {0: 'LVLALD', 1: 'LVLAHD', 2: 'LVHALD', 3: 'LVHAHD', 
                 4: 'HVLALD', 5: 'HVLAHD', 6: 'HVHALD', 7: 'HVHAHD'}
    df['eight_class_label'] = eight_class_labels.map(label_map)
        
    #Normalise and decimate
    if para.normalise:
        epochs = epochs.apply_function(lambda x: zscore(x, axis=-1))
    epochs.decimate(decim=int(para.eegFs / para.ds))
    
    #Group and Save by High/Low Conditions
    for idx, condition in enumerate(["valence", "arousal", "dominance"]):
        dim_dir = Path(para.output_dir)/ condition 
        for level, label in zip(["high", "low"], ["H", "L"]):
            condition_label = f"{label}{condition[0].upper()}"  #HV, HA, HD, LV, LA, LD
            condition_dir = dim_dir / condition_label        
            condition_dir.mkdir(parents=True, exist_ok=True)
            
            try:
            #Filter data for this level
                concatenated_data = np.concatenate([epoch[np.newaxis, :] for epoch, label in zip(epochs.get_data(),
                                                   labels) if label[idx] == (1 if level == "high" else 0)], axis=-1)
             
                concatenated_data = np.squeeze(concatenated_data, axis=0)
            
                #Save concatenated data
                file_name = f"sub-{subject}_label-{condition_label}.fif"
                file_path = condition_dir / file_name
                mne.io.RawArray(concatenated_data, epochs.info).save(file_path, overwrite=True)
    
                #Add rows to CSV
                if condition == "valence":
                    csv_rows_val.append({"id": f"sub-{subject}_label-{condition_label}",
                                         "file_path": str(file_path),
                                         "label": condition_label})
                
                elif condition == "arousal":                
                    csv_rows_aro.append({"id": f"sub-{subject}_label-{condition_label}",
                                         "file_path": str(file_path),
                                         "label": condition_label})
                
                elif condition == "dominance": 
                    csv_rows_dom.append({"id": f"sub-{subject}_label-{condition_label}",
                                         "file_path": str(file_path),
                                         "label": condition_label})
                
            except:
                continue

    #Extract eight-class labels as a NumPy array (matching trial order)
    epoch_labels = df['eight_class_label'].values
    
    #Define possible label combinations (multiclass)
    label_combinations = ['LVLALD', 'LVLAHD', 'LVHALD', 'LVHAHD', 
                          'HVLALD', 'HVLAHD', 'HVHALD', 'HVHAHD']
    
    #Iterate over each label combination
    for label_combo in label_combinations:
        #Create output directory for this label
        label_dir = Path(para.output_dir) / "VAD" / label_combo
        label_dir.mkdir(parents=True, exist_ok=True)
    
        #Filter data for this specific label combination
        selected_epochs = [epoch[np.newaxis, :] for epoch, label in
                           zip(epochs.get_data(), epoch_labels) if label == label_combo]

        if not selected_epochs:
            print(f"No epochs found for label {label_combo}, skipping...")
            continue  #Skip if no epochs match the label

        concatenated_data = np.concatenate(selected_epochs, axis=-1)
        concatenated_data = np.squeeze(concatenated_data, axis=0)

        #Save concatenated data
        file_name = f"sub-{subject}_label-{label_combo}.fif"
        file_path = label_dir / file_name
        mne.io.RawArray(concatenated_data, epochs.info).save(file_path, overwrite=True)

        #Add row to CSV
        csv_rows_VAD.append({"id": f"sub-{subject}_label-{label_combo}",
                             "file_path": str(file_path),
                             "label": label_combo})

    print(f"Processed and saved data for subject {subject}")

#Save CSVs
csv_output_path_val = Path(para.output_dir) / "dataset_val.csv"
pd.DataFrame(csv_rows_val).to_csv(csv_output_path_val, index=False)
print(f"Dataset CSV saved to {csv_output_path_val}")

csv_output_path_aro = Path(para.output_dir) / "dataset_aro.csv"
pd.DataFrame(csv_rows_aro).to_csv(csv_output_path_aro, index=False)
print(f"Dataset CSV saved to {csv_output_path_aro}")

csv_output_path_dom = Path(para.output_dir) / "dataset_dom.csv"
pd.DataFrame(csv_rows_dom).to_csv(csv_output_path_dom, index=False)
print(f"Dataset CSV saved to {csv_output_path_dom}")

csv_output_path = Path(para.output_dir) / "dataset_VAD.csv"
pd.DataFrame(csv_rows_VAD).to_csv(csv_output_path, index=False)
print(f"Dataset CSV saved to {csv_output_path}")