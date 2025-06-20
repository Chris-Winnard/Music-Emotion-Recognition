from scipy.stats import binom

def critical_accuracy_threshold(N, p_chance, alpha=0.05):
    """
    Finds the smallest k such that P(K ≥ k) ≤ alpha
    using the inverse survival function (isf).
    """
    #binom.isf(q, N, p) returns the smallest x with P(X > x) ≤ q
    x = binom.isf(alpha, N, p_chance)
    k = int(x) + 1  #Because P(X ≥ k) = P(X > k-1) = SF(k-1)
    return k, k / N  #critical count and corresponding accuracy threshold

def dataset_p_values_calculator(datasetInfo_dict, alpha=0.05):
    print(f"{datasetInfo_dict['Name']}:")
    N = datasetInfo_dict["N"]
    for key in ["Valence p_chance", "Aro p_chance", "Dom p_chance", "VAD p_chance"]:
        p_chance = datasetInfo_dict[key]
        k_critical, accuracy_critical = critical_accuracy_threshold(N, p_chance, alpha)
        print(f"{key} p={alpha} threshold accuracy: {accuracy_critical:.9f}")
 
    
###############################################################################################
"""Chance probability values ("p_chance") found from Σp_i^2 for each class i, for each given 
dataset/classification problem.

N = no. participants * no. trials * sub-trial epochs per trial
DEAP: N = 32*40*60 = 76800   
DAAMEE-S: 31*16*28 = 13888
DAAMEE-C: 29*16*28 = 12992"""

alpha = 0.05  #Significance level

DEAP_info_dict = {"Name":"DEAP",
                  "N":76800,
                  "Valence p_chance":0.508613281,
                  "Aro p_chance":0.515864258,
                  "Dom p_chance":0.529327393,
                  "VAD p_chance":0.176531982}

dataset_p_values_calculator(DEAP_info_dict, alpha)
print("\n")

DAAMEE_S_info_dict = {"Name":"DAAMEE-s",
                  "N":13888,
                  "Valence p_chance":0.505057088,
                  "Aro p_chance":0.504682622,
                  "Dom p_chance":0.501730104,
                  "VAD p_chance":0.15620623}

dataset_p_values_calculator(DAAMEE_S_info_dict, alpha)
print("\n")

DAAMEE_C_info_dict = {"Name":"DAAMEE-c",
                  "N":12992,
                  "Valence p_chance":0.50816502,
                  "Aro p_chance":0.509794321,
                  "Dom p_chance":0.501088258,
                  "VAD p_chance":0.156301816}

dataset_p_values_calculator(DAAMEE_C_info_dict, alpha)