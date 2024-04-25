"""This is an alternative evaluation method that uses softmax to select arms"""

from data_loading import parquet_reader
import numpy as np
import pandas as pd
import math
from collections import deque
import data_loading.deque as deque

def eval(test_df):
    df = test_df
                          
    #Number of rounds in test dataset
    t = len(df)

    #Retrieving arm scores from training data
    arm_score = {'B': -0.04973567168725407, 'A': 0.0037637496871528726, 'J': -0.027242881857316104, 'L': -0.02756762950836151, 'F': -0.04986739087941503, 'E': -0.048617719100845946, 'G': -0.03723720523271864, 'H': -0.05287368000216953, 'D': -0.03724869501480979, 'C': -0.5828979325264811, 'K': -0.028984839358533392}

    #Assigning each arm an equal random chance to be chosen, the old policy
    decay_arm_score = arm_score.copy()

    #Defining selection probabilities
    soft_max_prob = {}
    total_sum = 0
    old_sum = 0

    #Setting the arm exploration parameter
    explore = 0.9

    for index, row in df.iterrows():
        arm = row['selected_template']

        #Getting days since arm a last chosen from history column to retrieve d (days) for recency penalty
        for i in row['history'][::-1]:
                #not in training data
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