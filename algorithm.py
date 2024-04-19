import datareader
import numpy as np
import pandas as pd
import math
import ast
from collections import deque
import deque

def alg():
    """This function returns the scores for each arm template"""

    df = datareader.DataReader('data/train1.snappy.parquet', 500, 30000).read()

    #Defining dictionaries and deque
    arm_selected_dict = {}
    arm_eligible_dict = deque.LimitedDict(max_values=1500)

    #Stores each arm's score
    arm_score = {}

    #Getting unique languages
    unique_lang = []
    for i in df['ui_language'].unique():
        unique_lang.append(i)

    # Get unique values from the concatenated array
    unique_arms = []
    for i in df['selected_template'].unique():
        for j in i:
            if j.isalnum():
                unique_arms.append(j)
    unique_arms = list(set(unique_arms))


    #Initializing all arm scores as equal to begin
    for i in unique_arms:
        arm_score[i] = 0
        arm_eligible_dict.add_key(i)
        arm_selected_dict[i] = []

        subset = df.iloc[: :]


    #Iterating through each unique round [t], which is same as each row in dataframe
    for index, row in subset.iterrows():
        lang = row['ui_language']
        print(index)

        #Selecting arm 
        arm =   row['selected_template']
        
        #Store row indices for selected arm
        arm_selected_dict[arm].append(index)

        #Add all elgible values to eligible dict
        for eligible in row['eligible_templates']:
                arm_eligible_dict.add_value(eligible, index)

        # Get all previous rows where arm selected
        rows_with_arm = df.loc[arm_selected_dict[arm]]
        sel = len((rows_with_arm[rows_with_arm['session_end_completed'] == 1]))
        nonsel = len(rows_with_arm)

        if sel == 0:
            sel = 0.0001
        if nonsel == 0:
            nonsel = 0.0001

        mu_plus = sel / nonsel

        #Filter rows where arm is eligible but not selected
        rows_with_eligible = df.loc[arm_eligible_dict.get_values(arm)]
        not_selected_indices = list(set(rows_with_eligible.index) - set(rows_with_arm.index))
        rows_eligible_not_selected = df.loc[not_selected_indices]

        sel2 = len((rows_eligible_not_selected[rows_eligible_not_selected['session_end_completed'] == 1]))
        nonsel2 = len(rows_eligible_not_selected)

        if sel2 == 0:
            sel2 = 0.0001
        if nonsel2 == 0:
            nonsel2 = 0.0001
   
        mu_minus = sel2 / nonsel2

        #Handling divide by 0 cases
        if mu_plus == 0:
            mu_plus = 0.0001
        if (mu_minus == 0) or (pd.isna(mu_minus)):
            mu_minus = 0.0001

        #Relative difference calculation
        score = ((mu_plus-mu_minus)/mu_minus)

        #Getting days since arm a last chosen from history column to retrieve d for decay penalty
        for i in row['history'][::-1]:
            if i['template'] == arm:
                days = i['n_days']
                break
            else:
                days = 34
                
        #Adding decay function to score for recency effect
        score = score - (0.017*0.5)**(days/15)

        #Updating arm score
        arm_score[arm] = score
        
    return arm_score, unique_arms   


def original_score():
    df = datareader.DataReader('data/test1.snappy.parquet', 500, 10000).read()

    #Number of rounds in test dataset
    t = len(df)

    #Because baseline policy selects arms randomly from uniform distribution, simply
    #Measure number of times user completed lesson on time
    sel = len((df[df['session_end_completed'] == 1]))

    #Measure reward of this baseline policy -> 1/number of rounds * number of successes
    reward = ((1/t)* sel)

    return reward


def newpolicy_score():
    #Reading in data
    df = datareader.DataReader('data/test1.snappy.parquet', 500, 10000).read()

    #Number of rounds in test dataset
    t = len(df)

    #Retrieving arm scores from training data
    arm_score, unique = alg()
    filtered_dict = {key: value for key, value in arm_score.items() if value != 0}

    #Defining selection probabilities
    soft_max_prob = {}

    # Computing softmax of arm scores [π (a|t)]
    for eligible in filtered_dict.keys():
        soft_max_prob[eligible] = math.exp(filtered_dict[eligible])

    # Assigning each arm an equal random chance to be chosen, the old policy
    old_policy = {}
    old_policy = {arm: 1/len(unique) for arm in unique}

    sum_probs = sum(soft_max_prob.values())

    # Probability for each arm on custom algorithm
    new_policies = {arm: soft_max_prob[arm] / sum_probs for arm in soft_max_prob}
    
    diff = {key: new_policies[key] / old_policy.get(key, 0) for key in new_policies}

    # Apply the multiplication based on conditions
    total_sum = 0

    # Iterate through the DataFrame and calculate results, multiplying new reward by difference in policies
    for index, row in df.iterrows():
        if row['session_end_completed'] == 1:
            template = row['selected_template']
            value_to_multiply = diff.get(template, 0)
            result = row['session_end_completed'] * value_to_multiply
            df.at[index, 'result'] = result
            total_sum += result

    new_reward = ((1/t)* total_sum)
    original_reward = original_score()

    rel_diff = ((new_reward - original_reward) / ((new_reward + original_reward)/2)) * 100

    return rel_diff

if __name__ == '__main__':
    newpolicy_score()