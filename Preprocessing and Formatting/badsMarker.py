import pathlib
import pandas as pd
import mne
"If choosing to remove channels from DAAMEE which were thought to be noisier."

def badsMarker(eeg_raw, sub, eeg_type, task):
    lowerQualityChannels = f'./bids_dataset/derivatives/lowerQualityChannels.tsv'
    
    df = pd.read_csv(lowerQualityChannels, sep='\t')
    columnOfInterest = "Chans to remove (eeg_type=" + eeg_type + ", task=" + task + "):"
    
    if columnOfInterest not in df.columns:
        raise ValueError(f"Column {columnOfInterest} does not exist in the Excel file")
    
    df = df[["sub:", columnOfInterest]]
    
    row = df[df["sub:"] == sub] #Find the row for the given participant
    
    if row.empty:
        raise ValueError(f"No data found for participant {sub}")
    
    badsList = row[columnOfInterest].values[0] #Extract the channel list from the row
    
    if pd.isna(badsList) or badsList == "None":
        badsList = []
    else:
        badsList = [chan.strip() for chan in badsList.split(",")]

    #Filter to include only channels present in eeg_raw. E.g., Fpz may have been removed from ceegrid data
    existing_channels = eeg_raw.info["ch_names"]
    badsList = [chan for chan in badsList if chan in existing_channels]
    
    eeg_raw.info["bads"] = badsList
    
    return eeg_raw