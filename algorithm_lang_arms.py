from data_loading import parquet_reader
import numpy as np
import pandas as pd
import math
from collections import deque
import data_loading.deque as deque

def alg2(train_df):
    """This function returns the scores for each arm template, using the language +
        each notification to create more unique arms
    """
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

    lang = df['ui_language'].unique()

    #Get unique arms from the concatenated array
    unique_arms = []
    for i in df['selected_template'].unique():
        for j in lang:
            unique_arms.append((i, j))
 

        #Initializing all arm scores as equal to begin
        arm_score[i] = 0
        indecies=df[df['selected_template'] == i].index.to_list()
        arm_selected_dict[i] = indecies

    rows_with_arm = {}
    for arm in unique_arms:
        rows_with_arm[arm] = (df[(df['selected_template'] == arm[0]) & (df['ui_language'] == arm[1])]).index

    not_sel_dict = {}
    
    for i in unique_arms:
            arm_eligible_dict.add_key(i)
            arm_eligible_dict.add_value(i, (df[(df['selected_template'] == arm[0]) & (df['ui_language'] == arm[1])]).index) 

            rows_with_eligible = df.loc[arm_eligible_dict.get_values(i)[0]]
            not_selected_indices = list(set(rows_with_eligible.index) - set(rows_with_arm[i]))
            not_sel_dict[i] = not_selected_indices

    #Calculating Mu_Plus for each arm
    for i in unique_arms:
        arm = df[(df['selected_template'] == i[0]) & (df['ui_language'] == i[1])]
        average_selected_reward = len(arm[arm['session_end_completed'] == 1])
        length_hist_rounds = len(arm)

        if length_hist_rounds == 0:
             length_hist_rounds = 1

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


def eval(test_df, train_df):
    df = test_df
    #Number of rounds in test dataset
    t = len(df)

    #Retrieving arm scores from training data
    arm_score, _ =  alg2(train_df)

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
                    decay_arm_score[(i['template'], row['ui_language'])] = arm_score[i['template'], row['ui_language']] - (0.017*0.5)**(days/15)

        # Computing softmax of arm scores [π (a|t)]
        new_policies = {}
        # Initialize dictionaries
        soft_max_prob = {}
        old_policy = {}

        # Calculate old policy and softmax probabilities
        for eligible in row['eligible_templates']:
            # Choosing at random from uniform distribution
            old_policy[eligible] = (1 / len(row['eligible_templates']))
            
            soft_max_prob[eligible] = math.exp(decay_arm_score[eligible] / explore)

        # Calculate new policy using argmax
        arg_max_arm = max(soft_max_prob, key=soft_max_prob.get)
        new_policies = {arm: 1.0 if arm == arg_max_arm else 0.0 for arm in soft_max_prob}
    
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
    eval(parquet_reader.DataReader('data/test1.snappy.parquet', 5000, 500000).read(),
         parquet_reader.DataReader('data/train1.snappy.parquet', 5000, 2000000).read())

    