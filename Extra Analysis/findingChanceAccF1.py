import os
import pandas as pd
import numpy as np
import pickle

print("Chance acc and F1 for random classifier using proportions of each class.")

###########################################################################################################
#DAAMEE:

bids_folder = f"./bids_dataset"

#Function to load behavioral data
def load_bids_labels(bids_folder, exclude_participants):
    label_counts = {"combined_label": {}}  #Store all labels in a unified structure

    for sub in os.listdir(bids_folder):
        if not sub.startswith("sub-") or sub in exclude_participants:
            continue  #Skip non-subject folders and excluded subjects

        behav_file = os.path.join(bids_folder, sub, "beh", f"{sub}_task-emotion_beh.tsv")
        if not os.path.exists(behav_file):
            continue  #Skip if the file doesn't exist

        df = pd.read_csv(behav_file, sep="\t")

        #Binarize labels: high (>=5) -> 1, low (<5) -> 0
        df["valence"] = (df["valenceResp.response"] >= 5).astype(int)
        df["arousal"] = (df["arousalResp.response"] >= 5).astype(int)
        df["dominance"] = (df["dominanceResp.response"] >= 5).astype(int)

        #Combine labels into an 8-class label
        df["combined_label"] = (df["valence"].astype(str) + "_" + 
                                df["arousal"].astype(str) + "_" + 
                                df["dominance"].astype(str))

        #Initialize participant dictionary
        label_counts["combined_label"][sub] = {"valence": df["valence"].value_counts().to_dict(),
                                               "arousal": df["arousal"].value_counts().to_dict(),
                                               "dominance": df["dominance"].value_counts().to_dict(),
                                               "combined": df["combined_label"].value_counts().to_dict()}

    return label_counts

def compute_chance_accuracy_f1(label_counts):
    all_classes = ['0_0_0', '0_0_1', '0_1_0', '0_1_1', '1_0_0', '1_0_1', '1_1_0', '1_1_1']

    #Initialize total counts across all participants
    total_counts = {"valence": {0: 0, 1: 0}, "arousal": {0: 0, 1: 0},
                    "dominance": {0: 0, 1: 0}, "combined": {cls: 0 for cls in all_classes}}
    total_samples = 0

    #Aggregate class counts across all participants
    for participant, counts in label_counts["combined_label"].items():
        for key in ["valence", "arousal", "dominance"]:
            for cls, count in counts[key].items():
                total_counts[key][cls] += count
        for cls, count in counts["combined"].items():
            total_counts["combined"][cls] += count
        total_samples += sum(counts["combined"].values())

    #Compute class probabilities based on proportions
    class_probs = {key: [count / total_samples for count in total_counts[key].values()] 
                   for key in total_counts}
    print("Class probs:", class_probs)
    
    #Compute chance accuracy: sum(p^2) over all classes
    chance_acc = {key: sum(p**2 for p in probs) for key, probs in class_probs.items()}

    #Assign fixed F1 values
    chance_f1 = {"valence": 0.5, "arousal": 0.5, "dominance": 0.5, "combined": 0.125}

    return {key: {"chance_accuracy": chance_acc[key], "chance_f1": chance_f1[key]}
            for key in ["valence", "arousal", "dominance", "combined"]}

###########################################################################################################
#DAAMEE-s:
    
exclude_participants_DAAMEE_s = {"sub-04"}
DAAMEE_s_labels = load_bids_labels(bids_folder, exclude_participants_DAAMEE_s)
DAAMEE_s_chanceMetrics = compute_chance_accuracy_f1(DAAMEE_s_labels)

print("Chance values for DAAMEE-s:")
print(DAAMEE_s_chanceMetrics)

###########################################################################################################
#DAAMEE-c:
    
exclude_participants_DAAMEE_c = {"sub-04", "sub-16", "sub-21"}
DAAMEE_c_labels = load_bids_labels(bids_folder, exclude_participants_DAAMEE_c)
DAAMEE_c_chanceMetrics = compute_chance_accuracy_f1(DAAMEE_c_labels)

print("Chance values for DAAMEE-c:")
print(DAAMEE_c_chanceMetrics)

###########################################################################################################
#DEAP:

deap_folder = f'./data_preprocessed_python'

def load_deap_labels(deap_folder):
    label_counts = {"combined_label": {}}  #Store all labels in a unified structure

    for file in os.listdir(deap_folder):
        file_path = os.path.join(deap_folder, file)
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f, encoding="latin1")  #Use 'latin1' due to DEAP's format
            labels = data_dict["labels"]   #Shape: (40, 4)
            
        df = pd.DataFrame()

        #Binarize labels: High (>=5) -> 1, Low (<5) -> 0
        df["valence"] = (labels[:,0] >= 5).astype(int)
        df["arousal"] = (labels[:,1] >= 5).astype(int)
        df["dominance"] = (labels[:,2] >= 5).astype(int)

        #Combine labels into an 8-class label
        df["combined_label"] = (df["valence"].astype(str) + "_" + 
                                df["arousal"].astype(str) + "_" + 
                                df["dominance"].astype(str))

        #Initialize participant dictionary
        label_counts["combined_label"][file] = {"valence": df["valence"].value_counts().to_dict(),
                                                "arousal": df["arousal"].value_counts().to_dict(),
                                                "dominance": df["dominance"].value_counts().to_dict(),
                                                "combined": df["combined_label"].value_counts().to_dict()}

    return label_counts

#Load data
DEAP_labels = load_deap_labels(deap_folder)
DEAP_chanceMetrics = compute_chance_accuracy_f1(DEAP_labels)

print("Chance values for DEAP:")
print(DEAP_chanceMetrics)