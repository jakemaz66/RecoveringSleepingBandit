import parquet_reader
import numpy as np
import pandas as pd
import math
from collections import deque
import deque

def alg(train_df):
    """This function returns the scores for each arm template"""
    df = train_df

    #Defining dictionaries and deque with max values for historical arms
    arm_selected_dict = {}
    arm_eligible_dict = deque.LimitedDict(max_values=4000)

    #Stores each arm's score
    arm_score = {}

    #Get unique arms from the concatenated array
    unique_arms = []
    for i in df['selected_template'].unique():
        unique_arms.append(i)

        #Initializing all arm scores as equal (0) to begin
        arm_score[i] = 0
        arm_eligible_dict.add_key(i)
        arm_selected_dict[i] = []

    subset = df.iloc[:5000, :]

    #Iterating through each unique round [t], which is same as each row in dataframe
    for index, row in subset.iterrows():
        print(index)

        #Selecting arm 
        arm = row['selected_template']
        
        #Store row index for selected arm
        arm_selected_dict[arm].append(index)

        #Add all elgible values (the indicies where arm eligible) to eligible dict
        for eligible in row['eligible_templates']:
                arm_eligible_dict.add_value(eligible, index)

        #Get all previous rows where arm selected and apply reward function to calculate mu plus
        rows_with_arm = df.loc[arm_selected_dict[arm]]
        selected = len((rows_with_arm[rows_with_arm['session_end_completed'] == 1]))
        #Multiplying arm score by reward function
        selected = selected * arm_score[arm]

        #Get all previous rows where arm eligible but not selected and apply reward function to get mu minus
        rows_with_eligible = df.loc[arm_eligible_dict.get_values(arm)]

        #Rows where eligible
        total = len(rows_with_eligible)

        if (selected == 0):
            selected = 0.0001
        if (total == 0):
            total = 0.0001

        mu_plus = selected / total

        #C
        not_selected_indices = list(set(rows_with_eligible.index) - set(rows_with_arm.index))
        rows_eligible_not_selected = df.loc[not_selected_indices]

        selected_minus = len((rows_eligible_not_selected[rows_eligible_not_selected['session_end_completed'] == 1]))
        selected_minus = (1-arm_score[arm])**-1 * selected_minus

        total_minus = len(rows_eligible_not_selected)

        if selected_minus == 0:
            selected_minus = 0.0001
        if total_minus == 0:
            total_minus = 0.0001
   
        mu_minus = selected_minus / total_minus

        #Handling divide by 0 cases
        if (mu_plus == 0):
            mu_plus = 0.0001
        if (mu_minus == 0):
            mu_minus = 0.0001

        #Relative difference calculation to get the arm score
        score = ((mu_plus-mu_minus)/mu_minus)

        #Getting days since arm a last chosen from history column to retrieve d (days) for recency penalty
        for i in row['history'][::-1]:
            if i['template'] == arm:
                days = i['n_days']
                break
            else:
                #If arm not in history, set default days to 34
                days = 34
                
        #Adding decay function to score for recency effect
        score = score - (0.017*0.5)**(days/15)

        #Updating arm score in arm_score dictionary
        arm_score[arm] = score
        
    return arm_score, unique_arms   


def newpolicy_score(test_df, train_df):
    """This function returns the difference between the new policy and the old"""

    df = test_df

    #Number of rounds in test dataset
    t = len(df)

    #Retrieving arm scores from training data
    arm_score, unique = alg(train_df)
    filtered_dict = {key: value for key, value in arm_score.items() if value != 0}

    #Assigning each arm an equal random chance to be chosen, the old policy
    old_policy = {}
    old_policy = {arm: 1/len(unique) for arm in unique}

    #Defining selection probabilities
    soft_max_prob = {}

    # Computing softmax of arm scores [π (a|t)], add an explore hyperparameter to divide by
    explore=1.1
    for eligible in filtered_dict.keys():
        soft_max_prob[eligible] = math.exp(filtered_dict[eligible]/explore)

    #Calculating new policy
    divided_values = [value / explore for value in soft_max_prob.values()]
    sum_probs = sum(divided_values)

    new_policies = {arm: soft_max_prob[arm] / sum_probs for arm in soft_max_prob}
    
    #Getting differences in policy weights for each arm
    diff = {key: new_policies[key] / old_policy.get(key, 0) for key in new_policies}

   #Iterate through the DataFrame and calculate results, multiplying new reward by difference in policies (new weights)
    total_sum = 0
    for index, row in df.iterrows():
        if row['session_end_completed'] == 1:
            template = row['selected_template']
            value_to_multiply = diff.get(template, 0)
            result = row['session_end_completed'] * value_to_multiply
            df.at[index, 'result'] = result
            total_sum += result

    #Calculating reward of new behavior policy
    new_reward = ((1/t)* total_sum)

    #Measure reward of this baseline policy -> 1/number of rounds * number of successes
    selected = len((df[df['session_end_completed'] == 1]))
    original_reward = ((1/t)* selected)

    #Measuring the difference between new and old behavior policy
    rel_diff = ((new_reward - original_reward) / ((new_reward + original_reward)/2)) * 100

    return rel_diff

if __name__ == '__main__':
    newpolicy_score(parquet_reader.DataReader('data/test1.snappy.parquet', 500, 5000).read(),
                    parquet_reader.DataReader('data/train1.snappy.parquet', 500, 100000).read())