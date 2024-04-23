import parquet_reader
import numpy as np
import pandas as pd
import math
from collections import deque
import deque
#Onehot encode eligible
#np.cumsum()
#solve with pyspark 


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

        #Initializing all arm scores as equal to begin
        arm_score[i] = 0
        arm_eligible_dict.add_key(i)
        arm_selected_dict[i] = []


    subset = df.iloc[: :]

    #Iterating through each unique round [t], which is same as each row in dataframe
    for index, row in subset.iterrows():
        print(index)

        #Selecting arm and adding arm probability to historical rounds where arm selected
        arm = row['selected_template']
        
        #Store row index for selected arm
        arm_selected_dict[arm].append(index)

        #Add all elgible values (the indicies where arm eligible) to eligible dict
        for eligible in row['eligible_templates']:
                arm_eligible_dict.add_value(eligible, index)

        rows_with_arm = df.loc[arm_selected_dict[arm]]
        #Reward function application
        selected = len((rows_with_arm[rows_with_arm['session_end_completed'] == 1]))

        mu_plus = selected / len(rows_with_arm)

        #Get all previous rows where arm eligible but not selected and apply reward function to get mu minus
        rows_with_eligible = df.loc[arm_eligible_dict.get_values(arm)]

        not_selected_indices = list(set(rows_with_eligible.index) - set(rows_with_arm.index))
        rows_eligible_not_selected = df.loc[not_selected_indices]

        #Reward function application
        selected_minus = len((rows_eligible_not_selected[rows_eligible_not_selected['session_end_completed'] == 1]))

        leng = len(rows_eligible_not_selected)

        if (len(rows_eligible_not_selected) == 0):
            leng = 1
 
        mu_minus = selected_minus / leng

        #Handling divide by 0 cases
        if (mu_minus == 0):
            mu_minus = 1

        #Relative difference calculation to get the arm score
        score = ((mu_plus-mu_minus)/mu_minus)

        #Updating arm score in arm_score dictionary
        arm_score[arm] = score
 
    return arm_score, unique_arms  

 

def eval(test_df, train_df):
    df = test_df
    #Number of rounds in test dataset
    t = len(df)

    #Retrieving arm scores from training data
    arm_score, unique = alg(train_df)

    #Assigning each arm an equal random chance to be chosen, the old policy
    decay_arm_score = arm_score.copy()

    #Defining selection probabilities
    soft_max_prob = {}
    total_sum = 0

    subset = df.iloc[:, :]

    for index, row in df.iterrows():
        arm = row['selected_template']

        #Getting days since arm a last chosen from history column to retrieve d (days) for recency penalty
        for i in row['history'][::-1]:
            if i['template'] == arm:
                days = i['n_days']
                break
            else:
                #If arm not in history, set default days to 34
                days = 34
                
        #Adding decay function to score for recency effect, if this arm was recently chosen we want to lower it's prob for the round
        decay_arm_score[arm] = arm_score[arm] - (0.017*0.5)**(days/15)

        # Computing softmax of arm scores [π (a|t)]
        new_policies = {}
        soft_max_prob = {}
        old_policy = {}

        for eligible in row['eligible_templates']:
            old_policy[eligible] = (1/len(row['eligible_templates']))
            soft_max_prob[eligible] = math.exp(decay_arm_score[eligible])

            #Calculating new policy
            sum_probs = sum(soft_max_prob.values())
            new_policies = {arm: soft_max_prob[arm] / sum_probs for arm in soft_max_prob}
    
        #Getting differences in policy weights for each arm
        diff = {key: new_policies[key] / old_policy.get(key, 0) for key in new_policies}

        if row['session_end_completed'] == 1:
            value_to_multiply = diff.get(arm, 0)
            result = row['session_end_completed'] * value_to_multiply
            df.at[index, 'result'] = result
            total_sum += result

        decay_arm_score = arm_score.copy()

    #Calculating reward of new behavior policy
    new_reward = ((1/t)* total_sum)

    #Measure reward of this baseline policy -> 1/number of rounds * number of successes
    selected = len((df[df['session_end_completed'] == 1]))
    original_reward = ((1/t)* selected)

    #Measuring the difference between new and old behavior policy
    rel_diff = ((new_reward - original_reward) / ((new_reward + original_reward)/2)) * 100

    return rel_diff


if __name__ == '__main__':
    eval(parquet_reader.DataReader('data/test1.snappy.parquet', 200, 1400).read(),
                    parquet_reader.DataReader('data/train1.snappy.parquet', 500, 1001).read())