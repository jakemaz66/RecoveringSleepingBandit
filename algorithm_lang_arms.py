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


def eval(test_df):
    df = test_df
    #Number of rounds in test dataset
    t = len(df)

    #Retrieving arm scores from training data
    arm_score =  {('B', 'en'): 0.5076096206386886, ('B', 'es'): 0.181621535010415, ('B', 'pt'): 0.147608293984808, 
                 ('B', 'ar'): -0.13999999999999993, ('B', 'fr'): 0.4611224489795919, ('B', 'pl'): 1.1120141342756182, 
                 ('B', 'de'): 1.0187022900763358, ('B', 'ja'): 1.9334745762711865, ('B', 'ro'): 0.17903225806451606, 
                 ('B', 'it'): 0.3346448087431694, ('B', 'ru'): 0.3450046685340803, ('B', 'ko'): 0.5636363636363637, 
                 ('B', 'hu'): 0.5891304347826086, ('B', 'zs'): 0.6866379310344827, ('B', 'vi'): 0.412130177514793, 
                 ('B', 'cs'): 0.6786096256684493, ('B', 'tr'): -0.3935897435897436, ('B', 'dn'): 0.2473282442748091, 
                 ('B', 'id'): -0.312, ('B', 'hi'): -0.42115384615384616, ('B', 'uk'): 0.3013157894736842, ('B', 'el'): 0.0, 
                 ('B', 'th'): 0.6125, ('A', 'en'): 1.8745205479452058, ('A', 'es'): 1.2217112077895174, 
                 ('A', 'pt'): 1.2804079441760603, ('A', 'ar'): 0.49009900990099003, ('A', 'fr'): 1.8600461893764433, 
                 ('A', 'pl'): 2.0714285714285716, ('A', 'de'): 1.8559701492537317, ('A', 'ja'): 2.9065359477124186, 
                 ('A', 'ro'): 1.5150943396226415, ('A', 'it'): 2.004109589041096, ('A', 'ru'): 1.391715976331361, 
                 ('A', 'ko'): 2.6384615384615384, ('A', 'hu'): 2.6685314685314685, ('A', 'zs'): 2.1307017543859654, 
                 ('A', 'vi'): 1.8210191082802547, ('A', 'cs'): 1.7922077922077924, ('A', 'tr'): 0.32636815920398005, 
                 ('A', 'dn'): 1.1500000000000001, ('A', 'id'): 1.2764705882352942, ('A', 'hi'): 0.3578947368421052, 
                 ('A', 'uk'): 1.7741935483870968, ('A', 'el'): 1.7642857142857147, ('A', 'th'): 0.6125, ('J', 'en'): 0.5340921507922054, 
                 ('J', 'es'): 0.2064748201438848, ('J', 'pt'): 0.2371557747636662, ('J', 'ar'): -0.26835820895522383, 
                 ('J', 'fr'): 0.5275154004106776, ('J', 'pl'): 0.831869918699187, ('J', 'de'): 0.8287030941408822, 
                 ('J', 'ja'): 2.095216400911162, ('J', 'ro'): 0.17851851851851858, ('J', 'it'): 0.6503198294243071, 
                 ('J', 'ru'): 0.5355729406350669, ('J', 'ko'): 0.682608695652174, ('J', 'hu'): 0.849146757679181, 
                 ('J', 'zs'): 0.8655097613882864, ('J', 'vi'): 0.4110325318246111, ('J', 'cs'): 1.3644204851752024, 
                 ('J', 'tr'): -0.17815699658703066, ('J', 'dn'): 0.4893854748603353, ('J', 'id'): -0.5871999999999999, 
                 ('J', 'hi'): -0.27731092436974786, ('J', 'uk'): 0.4425806451612904, ('J', 'el'): 0.24637681159420297,
                   ('J', 'th'): -1.0, ('L', 'en'): 0.5310532276330691, ('L', 'es'): 0.19696068183645588, ('L', 'pt'): 0.17217636400241593, 
                   ('L', 'ar'): -0.31687943262411344, ('L', 'fr'): 0.5476293103448276, ('L', 'pl'): 0.7057851239669423, 
                   ('L', 'de'): 0.7939317319848292, ('L', 'ja'): 2.0241758241758245, ('L', 'ro'): 0.14050179211469538, 
                   ('L', 'it'): 0.5347412882787751, ('L', 'ru'): 0.27319540229885053, ('L', 'ko'): 0.5034965034965034, 
                   ('L', 'hu'): 0.8654411764705883, ('L', 'zs'): 0.6125, ('L', 'vi'): 0.615680473372781, ('L', 'cs'): 0.5399449035812672, 
                   ('L', 'tr'): -0.21455108359133127, ('L', 'dn'): -0.25217391304347825, ('L', 'id'): -0.5637681159420289, 
                   ('L', 'hi'): -0.07857142857142861, ('L', 'uk'): 0.4807947019867549, ('L', 'el'): 0.5087719298245613, 
                   ('L', 'th'): -0.09473684210526319, ('F', 'en'): 0.504941938968404, ('F', 'es'): 0.17991414918603701, 
                   ('F', 'pt'): 0.12090669904877545, ('F', 'ar'): -0.17619738751814223, ('F', 'fr'): 0.52848928384736, 
                   ('F', 'pl'): 0.813509060955519, ('F', 'de'): 0.9134474327628362, ('F', 'ja'): 1.9519450800915334, 
                   ('F', 'ro'): 0.21083032490974732, ('F', 'it'): 0.33736501079913606, ('F', 'ru'): 0.40495049504950503, 
                   ('F', 'ko'): 0.6270270270270272, ('F', 'hu'): 0.8830065359477124, ('F', 'zs'): 0.7711252653927815, 
                   ('F', 'vi'): 0.4913294797687861, ('F', 'cs'): 0.8974212034383955, ('F', 'tr'): 0.025964912280701736, 
                   ('F', 'dn'): 0.019259259259259313, ('F', 'id'): -0.47480916030534354, ('F', 'hi'): -0.4068965517241379, 
                   ('F', 'uk'): 0.1803921568627452, ('F', 'el'): 0.07500000000000001, ('F', 'th'): -0.21818181818181814, 
                   ('E', 'en'): 0.5413831443184345, ('E', 'es'): 0.12206896551724145, ('E', 'pt'): 0.1644169246646028, 
                   ('E', 'ar'): -0.3140324963072378, ('E', 'fr'): 0.2818181818181818, ('E', 'pl'): 0.9952000000000001, 
                   ('E', 'de'): 0.8108213820078226, ('E', 'ja'): 1.5901176470588234, ('E', 'ro'): 0.51171875, ('E', 'it'): 0.4653887113951012, 
                   ('E', 'ru'): 0.40190657769304095, ('E', 'ko'): 0.21037037037037046, ('E', 'hu'): 0.7082191780821917, 
                   ('E', 'zs'): 0.7733924611973393, ('E', 'vi'): 0.6196286472148541, ('E', 'cs'): 1.0703703703703702, 
                   ('E', 'tr'): -0.18669001751313483, ('E', 'dn'): -0.015662650602409678, ('E', 'id'): -0.7171052631578947, 
                   ('E', 'hi'): 0.12786885245901647, ('E', 'uk'): 0.45683060109289614, ('E', 'el'): 0.09206349206349201, 
                   ('E', 'th'): -0.46249999999999997, ('G', 'en'): 0.5576442113755548, ('G', 'es'): 0.18990163934426235, 
                   ('G', 'pt'): 0.23055499495459125, ('G', 'ar'): -0.41908957415565345, ('G', 'fr'): 0.5002028397565923, 
                   ('G', 'pl'): 0.8079545454545456, ('G', 'de'): 0.9164794007490636, ('G', 'ja'): 1.8400000000000003, 
                   ('G', 'ro'): 0.1838056680161943, ('G', 'it'): 0.4690045248868779, ('G', 'ru'): 0.4576940744143316, 
                   ('G', 'ko'): 1.1500000000000001, ('G', 'hu'): 0.6365671641791044, ('G', 'zs'): 0.7123893805309736, 
                   ('G', 'vi'): 0.3911764705882354, ('G', 'cs'): 0.7102272727272727, ('G', 'tr'): -0.19108910891089106, 
                   ('G', 'dn'): 0.3929577464788733, ('G', 'id'): -0.2575539568345323, ('G', 'hi'): -0.2833333333333334, 
                   ('G', 'uk'): 0.21411764705882352, ('G', 'el'): 0.32307692307692315, ('G', 'th'): 0.10256410256410248, 
                   ('H', 'en'): 0.5357831919953335, ('H', 'es'): 0.14629524657867043, ('H', 'pt'): 0.16460176991150444, 
                   ('H', 'ar'): -0.3080459770114943, ('H', 'fr'): 0.4570247933884297, ('H', 'pl'): 0.7088852988691438, 
                   ('H', 'de'): 0.7644914756025867, ('H', 'ja'): 1.8666666666666667, ('H', 'ro'): 0.2285714285714285, ('H', 'it'): 0.6085015940488842, ('H', 'ru'): 0.3444849589790338, ('H', 'ko'): 1.195744680851064, ('H', 'hu'): 1.1152103559870548, ('H', 'zs'): 0.8248306997742665, ('H', 'vi'): 0.15452054794520553, ('H', 'cs'): 0.45318559556786703, ('H', 'tr'): -0.05135623869801086, ('H', 'dn'): 0.11156462585034019, ('H', 'id'): -0.3426751592356688, ('H', 'hi'): -0.1898550724637681, ('H', 'uk'): 0.33678756476683935, ('H', 'el'): -0.1103448275862069, ('H', 'th'): -0.11794871794871797, ('D', 'en'): 0.5145500044686746, ('D', 'es'): 0.14542984979226603, ('D', 'pt'): 0.1706993152106245, ('D', 'ar'): -0.2884450784593438, ('D', 'fr'): 0.5114571746384872, ('D', 'pl'): 0.8750819672131148, ('D', 'de'): 0.7563380281690142, ('D', 'ja'): 2.0952586206896555, ('D', 'ro'): 0.7837037037037037, ('D', 'it'): 0.5570526315789475, ('D', 'ru'): 0.5394564198687909, ('D', 'ko'): 0.7606299212598425, ('D', 'hu'): 0.48571428571428565, ('D', 'zs'): 0.8000000000000002, ('D', 'vi'): 0.4835579514824799, ('D', 'cs'): 0.9637795275590553, ('D', 'tr'): -0.3343653250773993, ('D', 'dn'): -0.25466666666666665, ('D', 'id'): -0.27030303030303027, ('D', 'hi'): -0.47192982456140353, ('D', 'uk'): 0.4893854748603353, ('D', 'el'): 0.460377358490566, ('D', 'th'): -0.6814814814814815, ('C', 'en'): 2.721039260969977, ('C', 'es'): 2.333303146932307, ('C', 'pt'): 2.167725409836066, ('C', 'ar'): 1.838834951456311, ('C', 'fr'): 2.604709576138148, ('C', 'pl'): 2.8192546583850935, ('C', 'de'): 2.783578431372549, ('C', 'ja'): 3.794425087108014, ('C', 'ro'): 2.354, ('C', 'it'): 3.1979661016949157, ('C', 'ru'): 2.343620689655173, ('C', 'ko'): 2.75632183908046, ('C', 'hu'): 2.802721088435374, ('C', 'zs'): 2.1625806451612903, ('C', 'vi'): 3.1557046979865775, ('C', 'cs'): 2.293617021276596, ('C', 'tr'): 2.025925925925926, ('C', 'dn'): 1.1500000000000001, ('C', 'id'): 1.5294117647058827, ('C', 'hi'): 0.5925925925925926, ('C', 'uk'): 2.463888888888889, ('C', 'el'): 2.909090909090909, ('C', 'th'): 3.3000000000000003, ('K', 'en'): 0.5088391147526385, ('K', 'es'): 0.19850253602769494, ('K', 'pt'): 0.1628710858072386, ('K', 'ar'): -0.30566572237960343, ('K', 'fr'): 0.5570074812967583, ('K', 'pl'): 1.0717504332755632, ('K', 'de'): 0.803225806451613, ('K', 'ja'): 1.9305122494432072, ('K', 'ro'): 0.12734082397003735, ('K', 'it'): 0.5862412761714855, ('K', 'ru'): 0.4210715893741558, ('K', 'ko'): 1.6581818181818182, ('K', 'hu'): 1.365695792880259, ('K', 'zs'): 0.5553191489361703, ('K', 'vi'): 0.33190921228304404, ('K', 'cs'): 0.7154255319148937, ('K', 'tr'): -0.024013157894736778, ('K', 'dn'): 0.4072727272727273, ('K', 'id'): -0.21818181818181814, ('K', 'hi'): -0.5433628318584071, ('K', 'uk'): 0.2218579234972677, ('K', 'el'): 0.1217391304347826, ('K', 'th'): -0.8837209302325582}

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
        soft_max_prob = {}
        old_policy = {}

        for eligible in row['eligible_templates']:
            #Choosing at random from uniform distribution
            old_policy[eligible] = (1/len((row['eligible_templates'], row['ui_language'])))

            soft_max_prob[eligible] = math.exp(decay_arm_score[(eligible, row['ui_language'])]/explore)

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
    alg2(parquet_reader.DataReader('data/train1.snappy.parquet', 5000, 10000000).read())

    

    #ARM SCORES AFTER 500,000 rows
    arm_score = {('B', 'en'): 0.5076096206386886, ('B', 'es'): 0.181621535010415, ('B', 'pt'): 0.147608293984808, 
                 ('B', 'ar'): -0.13999999999999993, ('B', 'fr'): 0.4611224489795919, ('B', 'pl'): 1.1120141342756182, 
                 ('B', 'de'): 1.0187022900763358, ('B', 'ja'): 1.9334745762711865, ('B', 'ro'): 0.17903225806451606, 
                 ('B', 'it'): 0.3346448087431694, ('B', 'ru'): 0.3450046685340803, ('B', 'ko'): 0.5636363636363637, 
                 ('B', 'hu'): 0.5891304347826086, ('B', 'zs'): 0.6866379310344827, ('B', 'vi'): 0.412130177514793, 
                 ('B', 'cs'): 0.6786096256684493, ('B', 'tr'): -0.3935897435897436, ('B', 'dn'): 0.2473282442748091, 
                 ('B', 'id'): -0.312, ('B', 'hi'): -0.42115384615384616, ('B', 'uk'): 0.3013157894736842, ('B', 'el'): 0.0, 
                 ('B', 'th'): 0.6125, ('A', 'en'): 1.8745205479452058, ('A', 'es'): 1.2217112077895174, 
                 ('A', 'pt'): 1.2804079441760603, ('A', 'ar'): 0.49009900990099003, ('A', 'fr'): 1.8600461893764433, 
                 ('A', 'pl'): 2.0714285714285716, ('A', 'de'): 1.8559701492537317, ('A', 'ja'): 2.9065359477124186, 
                 ('A', 'ro'): 1.5150943396226415, ('A', 'it'): 2.004109589041096, ('A', 'ru'): 1.391715976331361, 
                 ('A', 'ko'): 2.6384615384615384, ('A', 'hu'): 2.6685314685314685, ('A', 'zs'): 2.1307017543859654, 
                 ('A', 'vi'): 1.8210191082802547, ('A', 'cs'): 1.7922077922077924, ('A', 'tr'): 0.32636815920398005, 
                 ('A', 'dn'): 1.1500000000000001, ('A', 'id'): 1.2764705882352942, ('A', 'hi'): 0.3578947368421052, 
                 ('A', 'uk'): 1.7741935483870968, ('A', 'el'): 1.7642857142857147, ('A', 'th'): 0.6125, ('J', 'en'): 0.5340921507922054, 
                 ('J', 'es'): 0.2064748201438848, ('J', 'pt'): 0.2371557747636662, ('J', 'ar'): -0.26835820895522383, 
                 ('J', 'fr'): 0.5275154004106776, ('J', 'pl'): 0.831869918699187, ('J', 'de'): 0.8287030941408822, 
                 ('J', 'ja'): 2.095216400911162, ('J', 'ro'): 0.17851851851851858, ('J', 'it'): 0.6503198294243071, 
                 ('J', 'ru'): 0.5355729406350669, ('J', 'ko'): 0.682608695652174, ('J', 'hu'): 0.849146757679181, 
                 ('J', 'zs'): 0.8655097613882864, ('J', 'vi'): 0.4110325318246111, ('J', 'cs'): 1.3644204851752024, 
                 ('J', 'tr'): -0.17815699658703066, ('J', 'dn'): 0.4893854748603353, ('J', 'id'): -0.5871999999999999, 
                 ('J', 'hi'): -0.27731092436974786, ('J', 'uk'): 0.4425806451612904, ('J', 'el'): 0.24637681159420297,
                   ('J', 'th'): -1.0, ('L', 'en'): 0.5310532276330691, ('L', 'es'): 0.19696068183645588, ('L', 'pt'): 0.17217636400241593, 
                   ('L', 'ar'): -0.31687943262411344, ('L', 'fr'): 0.5476293103448276, ('L', 'pl'): 0.7057851239669423, 
                   ('L', 'de'): 0.7939317319848292, ('L', 'ja'): 2.0241758241758245, ('L', 'ro'): 0.14050179211469538, 
                   ('L', 'it'): 0.5347412882787751, ('L', 'ru'): 0.27319540229885053, ('L', 'ko'): 0.5034965034965034, 
                   ('L', 'hu'): 0.8654411764705883, ('L', 'zs'): 0.6125, ('L', 'vi'): 0.615680473372781, ('L', 'cs'): 0.5399449035812672, 
                   ('L', 'tr'): -0.21455108359133127, ('L', 'dn'): -0.25217391304347825, ('L', 'id'): -0.5637681159420289, 
                   ('L', 'hi'): -0.07857142857142861, ('L', 'uk'): 0.4807947019867549, ('L', 'el'): 0.5087719298245613, 
                   ('L', 'th'): -0.09473684210526319, ('F', 'en'): 0.504941938968404, ('F', 'es'): 0.17991414918603701, 
                   ('F', 'pt'): 0.12090669904877545, ('F', 'ar'): -0.17619738751814223, ('F', 'fr'): 0.52848928384736, 
                   ('F', 'pl'): 0.813509060955519, ('F', 'de'): 0.9134474327628362, ('F', 'ja'): 1.9519450800915334, 
                   ('F', 'ro'): 0.21083032490974732, ('F', 'it'): 0.33736501079913606, ('F', 'ru'): 0.40495049504950503, 
                   ('F', 'ko'): 0.6270270270270272, ('F', 'hu'): 0.8830065359477124, ('F', 'zs'): 0.7711252653927815, 
                   ('F', 'vi'): 0.4913294797687861, ('F', 'cs'): 0.8974212034383955, ('F', 'tr'): 0.025964912280701736, 
                   ('F', 'dn'): 0.019259259259259313, ('F', 'id'): -0.47480916030534354, ('F', 'hi'): -0.4068965517241379, 
                   ('F', 'uk'): 0.1803921568627452, ('F', 'el'): 0.07500000000000001, ('F', 'th'): -0.21818181818181814, 
                   ('E', 'en'): 0.5413831443184345, ('E', 'es'): 0.12206896551724145, ('E', 'pt'): 0.1644169246646028, 
                   ('E', 'ar'): -0.3140324963072378, ('E', 'fr'): 0.2818181818181818, ('E', 'pl'): 0.9952000000000001, 
                   ('E', 'de'): 0.8108213820078226, ('E', 'ja'): 1.5901176470588234, ('E', 'ro'): 0.51171875, ('E', 'it'): 0.4653887113951012, 
                   ('E', 'ru'): 0.40190657769304095, ('E', 'ko'): 0.21037037037037046, ('E', 'hu'): 0.7082191780821917, 
                   ('E', 'zs'): 0.7733924611973393, ('E', 'vi'): 0.6196286472148541, ('E', 'cs'): 1.0703703703703702, 
                   ('E', 'tr'): -0.18669001751313483, ('E', 'dn'): -0.015662650602409678, ('E', 'id'): -0.7171052631578947, 
                   ('E', 'hi'): 0.12786885245901647, ('E', 'uk'): 0.45683060109289614, ('E', 'el'): 0.09206349206349201, 
                   ('E', 'th'): -0.46249999999999997, ('G', 'en'): 0.5576442113755548, ('G', 'es'): 0.18990163934426235, 
                   ('G', 'pt'): 0.23055499495459125, ('G', 'ar'): -0.41908957415565345, ('G', 'fr'): 0.5002028397565923, 
                   ('G', 'pl'): 0.8079545454545456, ('G', 'de'): 0.9164794007490636, ('G', 'ja'): 1.8400000000000003, 
                   ('G', 'ro'): 0.1838056680161943, ('G', 'it'): 0.4690045248868779, ('G', 'ru'): 0.4576940744143316, 
                   ('G', 'ko'): 1.1500000000000001, ('G', 'hu'): 0.6365671641791044, ('G', 'zs'): 0.7123893805309736, 
                   ('G', 'vi'): 0.3911764705882354, ('G', 'cs'): 0.7102272727272727, ('G', 'tr'): -0.19108910891089106, 
                   ('G', 'dn'): 0.3929577464788733, ('G', 'id'): -0.2575539568345323, ('G', 'hi'): -0.2833333333333334, 
                   ('G', 'uk'): 0.21411764705882352, ('G', 'el'): 0.32307692307692315, ('G', 'th'): 0.10256410256410248, 
                   ('H', 'en'): 0.5357831919953335, ('H', 'es'): 0.14629524657867043, ('H', 'pt'): 0.16460176991150444, 
                   ('H', 'ar'): -0.3080459770114943, ('H', 'fr'): 0.4570247933884297, ('H', 'pl'): 0.7088852988691438, 
                   ('H', 'de'): 0.7644914756025867, ('H', 'ja'): 1.8666666666666667, ('H', 'ro'): 0.2285714285714285, ('H', 'it'): 0.6085015940488842, ('H', 'ru'): 0.3444849589790338, ('H', 'ko'): 1.195744680851064, ('H', 'hu'): 1.1152103559870548, ('H', 'zs'): 0.8248306997742665, ('H', 'vi'): 0.15452054794520553, ('H', 'cs'): 0.45318559556786703, ('H', 'tr'): -0.05135623869801086, ('H', 'dn'): 0.11156462585034019, ('H', 'id'): -0.3426751592356688, ('H', 'hi'): -0.1898550724637681, ('H', 'uk'): 0.33678756476683935, ('H', 'el'): -0.1103448275862069, ('H', 'th'): -0.11794871794871797, ('D', 'en'): 0.5145500044686746, ('D', 'es'): 0.14542984979226603, ('D', 'pt'): 0.1706993152106245, ('D', 'ar'): -0.2884450784593438, ('D', 'fr'): 0.5114571746384872, ('D', 'pl'): 0.8750819672131148, ('D', 'de'): 0.7563380281690142, ('D', 'ja'): 2.0952586206896555, ('D', 'ro'): 0.7837037037037037, ('D', 'it'): 0.5570526315789475, ('D', 'ru'): 0.5394564198687909, ('D', 'ko'): 0.7606299212598425, ('D', 'hu'): 0.48571428571428565, ('D', 'zs'): 0.8000000000000002, ('D', 'vi'): 0.4835579514824799, ('D', 'cs'): 0.9637795275590553, ('D', 'tr'): -0.3343653250773993, ('D', 'dn'): -0.25466666666666665, ('D', 'id'): -0.27030303030303027, ('D', 'hi'): -0.47192982456140353, ('D', 'uk'): 0.4893854748603353, ('D', 'el'): 0.460377358490566, ('D', 'th'): -0.6814814814814815, ('C', 'en'): 2.721039260969977, ('C', 'es'): 2.333303146932307, ('C', 'pt'): 2.167725409836066, ('C', 'ar'): 1.838834951456311, ('C', 'fr'): 2.604709576138148, ('C', 'pl'): 2.8192546583850935, ('C', 'de'): 2.783578431372549, ('C', 'ja'): 3.794425087108014, ('C', 'ro'): 2.354, ('C', 'it'): 3.1979661016949157, ('C', 'ru'): 2.343620689655173, ('C', 'ko'): 2.75632183908046, ('C', 'hu'): 2.802721088435374, ('C', 'zs'): 2.1625806451612903, ('C', 'vi'): 3.1557046979865775, ('C', 'cs'): 2.293617021276596, ('C', 'tr'): 2.025925925925926, ('C', 'dn'): 1.1500000000000001, ('C', 'id'): 1.5294117647058827, ('C', 'hi'): 0.5925925925925926, ('C', 'uk'): 2.463888888888889, ('C', 'el'): 2.909090909090909, ('C', 'th'): 3.3000000000000003, ('K', 'en'): 0.5088391147526385, ('K', 'es'): 0.19850253602769494, ('K', 'pt'): 0.1628710858072386, ('K', 'ar'): -0.30566572237960343, ('K', 'fr'): 0.5570074812967583, ('K', 'pl'): 1.0717504332755632, ('K', 'de'): 0.803225806451613, ('K', 'ja'): 1.9305122494432072, ('K', 'ro'): 0.12734082397003735, ('K', 'it'): 0.5862412761714855, ('K', 'ru'): 0.4210715893741558, ('K', 'ko'): 1.6581818181818182, ('K', 'hu'): 1.365695792880259, ('K', 'zs'): 0.5553191489361703, ('K', 'vi'): 0.33190921228304404, ('K', 'cs'): 0.7154255319148937, ('K', 'tr'): -0.024013157894736778, ('K', 'dn'): 0.4072727272727273, ('K', 'id'): -0.21818181818181814, ('K', 'hi'): -0.5433628318584071, ('K', 'uk'): 0.2218579234972677, ('K', 'el'): 0.1217391304347826, ('K', 'th'): -0.8837209302325582}