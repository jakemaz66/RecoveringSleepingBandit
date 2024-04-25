import parquet_reader
import numpy as np
import pandas as pd
import math
from collections import deque
import deque

def alg2(train_df):
    """This function returns the scores for each arm template"""
    from sklearn.preprocessing import MultiLabelBinarizer
    #Setting data fram equal to passed arg
    df = train_df

    #One Hot encoding 'eligible_templates' column, which contains values of lists
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('eligible_templates')),
                          columns=mlb.classes_,
                          index=df.index))

    #Initializing dictionary for eligible arms
    arm_eligible_dict = deque.LimitedDict(max_values=500000)

    #Stores each arm's score
    arm_score = {}
    arm_selected_dict = {}

    #Get unique arms from the concatenated array
    unique_arms = []
    for i in df['selected_template'].unique():
        unique_arms.append(i)

        #Initializing all arm scores as equal to begin
        arm_score[i] = 0
        indecies=df[df['selected_template'] == i].index.to_list()
        arm_selected_dict[i] = indecies

    rows_with_arm = {}
    for arm in unique_arms:
        rows_with_arm[arm] = df[df['selected_template'] == arm].index

    not_sel_dict = {}
    
    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J','K', 'L']
    for i in df.columns:
        if i in cols:
            arm_eligible_dict.add_key(i)
            arm_eligible_dict.add_value(i, df[df[i] == 1].index) 

            rows_with_eligible = df.loc[arm_eligible_dict.get_values(i)[0]]
            not_selected_indices = list(set(rows_with_eligible.index) - set(rows_with_arm[i]))
            not_sel_dict[i] = not_selected_indices

    #Calculating Mu_Plus for each arm
    for i in unique_arms:
        arm = df[df['selected_template'] == i]
        average_selected_reward = len(arm[arm['session_end_completed'] == 1])
        length_hist_rounds = len(arm)

        mu_plus = average_selected_reward / length_hist_rounds

        rows_eligible_not_selected =  not_sel_dict[i]
        
        eligible_df = df.loc[rows_eligible_not_selected]
        average_eligible_not_selected = len(eligible_df[eligible_df['session_end_completed'] == 1])
        length_hist_notsel_rounds = len(rows_eligible_not_selected)

        #Divide by 0 case for template 'C'
        if length_hist_notsel_rounds == 0:
            length_hist_notsel_rounds = 1

        mu_minus = average_eligible_not_selected / length_hist_notsel_rounds

        if mu_minus == 0:
            mu_minus = 1

        score = ((mu_plus-mu_minus)/mu_minus)

        arm_score[i] = score
 
    return arm_score, unique_arms 


#Arm scores after 500,000 rows 
#{'B': -0.047331391145729304, 'A': 0.006045519325268447, 'J': -0.01694617539841133, 'L': -0.04136915628296414, 'F': -0.04549318729108681, 'E': -0.05108588091778463, 'G': -0.023230439457632902, 'H': -0.05199452534002813, 'D': -0.04358103756884987, 'C': -0.5859359258512996, 'K': -0.038284107850491876}


def eval(test_df):
    df = test_df
    #Number of rounds in test dataset
    t = len(df)

    #Retrieving arm scores from training data
    arm_score = {'B': -0.047331391145729304, 'A': 0.006045519325268447, 'J': -0.01694617539841133, 'L': -0.04136915628296414, 'F': -0.04549318729108681, 'E': -0.05108588091778463, 'G': -0.023230439457632902, 'H': -0.05199452534002813, 'D': -0.04358103756884987, 'C': -0.5859359258512996, 'K': -0.038284107850491876}

    #Assigning each arm an equal random chance to be chosen, the old policy
    decay_arm_score = arm_score.copy()

    #Defining selection probabilities
    soft_max_prob = {}
    total_sum = 0
    old_sum = 0

    subset = df.iloc[:, :]

    #Setting the arm exploration parameter
    explore = 1.01

    for index, row in subset.iterrows():
        arm = row['selected_template']

        #Getting days since arm a last chosen from history column to retrieve d (days) for recency penalty
        #If arm is selcted recently, you want it to be lower prob in the softmax calculation -> always hurts new policy score find better implementation
        for i in row['history'][::-1]:
                if i['template'] == 'I':
                    break
                else:
                    days = i['n_days']
                    #Adding decay function to score for recency effect, if this arm was recently chosen we want to lower it's prob for the round
                    decay_arm_score[i['template']] = arm_score[i['template']] - (0.017*0.5)**(days/15)

        # Computing softmax of arm scores [π (a|t)]
        new_policies = {}
        soft_max_prob = {}
        old_policy = {}

        for eligible in row['eligible_templates']:
            #Choosing at random from uniform distribution
            old_policy[eligible] = (1/len(row['eligible_templates']))

            soft_max_prob[eligible] = math.exp(decay_arm_score[eligible]/explore)

            #Calculating new policy
            sum_probs = sum(soft_max_prob.values())
            new_policies = {arm: soft_max_prob[arm] / sum_probs for arm in soft_max_prob}
    
        #Getting differences in policy weights from old policy and new policy for each arm
        diff = {key: new_policies[key] / old_policy.get(key, 0) for key in new_policies}

        #If reward function is 1, apply weighted multiplication of new policy
        if row['session_end_completed'] == 1:
            old_sum += 1
            value_to_multiply = diff.get(arm, 0)
            result = row['session_end_completed'] * value_to_multiply
            df.at[index, 'result'] = result
            total_sum += result

        #Resetting arm scores to offset recency penalty
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
    eval(parquet_reader.DataReader('data/test1.snappy.parquet', 5000, 100000).read())