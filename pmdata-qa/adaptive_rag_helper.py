import json
import pandas as pd
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, f1_score
import math
import re

def load_data(path, expand):
    # loading sleep data
    df = pd.read_json(path + 'fitbit/sleep.json')
    df['dateTime'] = df['dateOfSleep'].apply(lambda x: x[:10])
    df_sleep = df[['dateTime', 'minutesAsleep']]

    # loading heart rate data
    if os.path.exists(path + 'fitbit/resting_heart_rate.json'):
        df = pd.read_json(path + 'fitbit/resting_heart_rate.json')
        df['heartRate'] = df['value'].apply(lambda x: x['value']).round(2)

        df['dateTime'] = df['dateTime'].astype(str)
        df_hr = df[['dateTime', 'heartRate']]
    else:
        df_hr = pd.DataFrame()
        df_hr.loc[:, 'dateTime'] = df_sleep.loc[:, 'dateTime']
        df_hr['heartRate'] = np.nan

    # loading step data
    with open(path + 'fitbit/steps.json', 'r') as file:
        step_data = json.load(file)

    if not expand:
        steps_per_day = defaultdict(int)
        for entry in step_data:
            dateTime = entry['dateTime']
            date = dateTime.split()[0]
            value = int(float(entry['value']))
            steps_per_day[date] += value

        df_steps = pd.DataFrame(list(steps_per_day.items()), columns=['dateTime', 'steps'])
    else:
        steps_per_day = defaultdict(lambda: defaultdict(int))
        for entry in step_data:
            dateTime = entry['dateTime']
            date = dateTime.split()[0]
            time = dateTime.split()[1][:2]
            value = int(float(entry['value']))
            steps_per_day[date][time]+=value

    df_steps = pd.DataFrame(list(steps_per_day.items()), columns=['dateTime', 'steps'])


    # loading calorie data
    with open(path + 'fitbit/calories.json', 'r') as file:
        calorie_data = json.load(file)

    if not expand:
        calories_per_day = defaultdict(int)
        for entry in calorie_data:
            dateTime = entry['dateTime']
            date = dateTime.split()[0]
            value = int(float(entry['value']))
            calories_per_day[date] += value
    else:
        calories_per_day = defaultdict(lambda: defaultdict(int))
        for entry in calorie_data:
            dateTime = entry['dateTime']
            date = dateTime.split()[0]
            time = dateTime.split()[1][:2]
            value = int(float(entry['value']))
            calories_per_day[date][time] += value

    df_calories = pd.DataFrame(list(calories_per_day.items()), columns=['dateTime', 'calories'])

    # loading wellness data
    df_wellness = pd.read_csv(path + 'pmsys/wellness.csv')
    df_wellness['dateTime'] = df_wellness['effective_time_frame'].apply(lambda x: x[:10])
    df_wellness_final = df_wellness[['dateTime', 'fatigue', 'mood', 'readiness']]
    df_wellness_final = df_wellness_final[df_wellness_final['fatigue'] != 0]

    df = pd.merge(df_hr, df_sleep, on='dateTime', how='outer')
    df = pd.merge(df, df_steps, on='dateTime', how = 'outer')
    df = pd.merge(df, df_calories, on='dateTime', how = 'outer')
    df = pd.merge(df, df_wellness_final, on = 'dateTime', how = 'outer')
    
    return df


def balance_samples(df, target, sample_size, bins = None):
    df_cleaned = df.copy().dropna(subset = [target])
    if bins:
        df_cleaned['bin'] = pd.cut(df_cleaned[target], bins)
        grouped = df_cleaned.groupby('bin', observed = False)
    else:
        grouped = df_cleaned.groupby(target)

    sampled_indices = []
    for group_name, group_data in grouped:
        group_indices = group_data.index.to_list()
        if len(group_data) > 0:
            if len(group_data) < sample_size:
                # upsample if needed
                np.random.seed(0)
                sampled_indices.extend(np.random.choice(group_indices, sample_size, replace=True))
            else:
                sampled_indices.extend(np.random.choice(group_indices, sample_size))

    return sampled_indices

def compute_statistics(data):
    data = np.array(data)
    if np.all(np.isnan(data)):
        stats = {
            'avg': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
          }
    else:
        stats = {
            'avg': int(np.nanmean(data)),
            'std': int(np.nanstd(data)),
            'min': int(np.nanmin(data)),
            'max': int(np.nanmax(data))
        }
    return stats

def create_plot_with_all_data(steps, heart_rate, calories, sleep, output_path):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)   #top left
    ax2 = fig.add_subplot(222)   #top right
    ax3 =fig.add_subplot(223)   #bottom left
    ax4 = fig.add_subplot(224)   #bottom right 
    fig.tight_layout()
    
    ax1.plot(steps, marker='o', color = 'black', linestyle='-')
    ax1.set_title('Number of Steps Per Day')
    ax2.plot(heart_rate, marker='o', color = 'black', linestyle='-')
    ax2.set_title('Resting Heart Rate (Beats per Minute)')
    ax3.plot(calories, marker='o', color = 'black', linestyle='-')
    ax3.set_title('Calories Burned Per Day (kcal)')
    ax4.plot(sleep, marker='o', color = 'black', linestyle='-')
    ax4.set_title('Minutes Asleep Per Night')

    plt.savefig(output_path + '/plot.png')

def create_multiple_plots(steps, heart_rate, calories, sleep, output_path):
    # steps plot 
    plt.figure()
    plt.plot(steps, marker='o', color = 'black', linestyle='-')
    plt.title('Number of Steps Per Day')
    plt.savefig(output_path + '/steps.png')
    plt.close()

    # heart rate plot 
    plt.figure()
    plt.plot(heart_rate, marker='o', color = 'black', linestyle='-')
    plt.title('Resting Heart Rate (Beats per Minute)')
    plt.savefig(output_path + '/heart_rate.png')
    plt.close()

    # calories plot 
    plt.figure()
    plt.plot(calories, marker='o', color = 'black', linestyle='-')
    plt.title('Calories Burned Per Day (kcal)')
    plt.savefig(output_path + '/calories.png')
    plt.close()

    # sleep plot 
    plt.figure()
    plt.plot(sleep, marker='o', color = 'black', linestyle='-')
    plt.title('Minutes Asleep Per Night')
    plt.savefig(output_path + '/sleep.png')
    plt.close()

def score_classification(labels, preds):
    preds_int = []
    labels_int = []
    for i, pred in enumerate(preds):
        try:
            if int(float(pred)) != math.nan:
                if int(float((pred))) > 4:
                    preds_int.append(4)
                else:
                    preds_int.append(int(float((pred))))
            else:
                preds_int.append(2)
            labels_int.append(int(float(labels[i])))
        except ValueError:
            preds_int.append(2)
            labels_int.append(int(float(labels[i])))


    acc = accuracy_score(labels_int, preds_int)
    print("accuracy", acc)
    macro_f1 = f1_score(labels_int, preds_int, average = 'macro')
    print("macro f1", macro_f1)
    micro_f1 = f1_score(labels_int, preds_int, average = 'micro')
    print("micro f1", micro_f1)

    return acc, micro_f1, macro_f1

def score_regression(labels, preds):
    preds_float = []
    labels_float = []
    for i, pred in enumerate(preds):
        try:
            if float(pred) + 1 <= 11:
                preds_float.append(float(pred) + 1) 
                labels_float.append(float(labels[i]) + 1)
            else:
                preds_float.append(11)
                labels_float.append(float(labels[i]) + 1)
        except ValueError:
            preds_float.append(6)
            labels_float.append(float(labels[i]) + 1)

    MAE = mean_absolute_error(labels_float, preds_float)
    print("MAE", MAE)

    MAPE = mean_absolute_percentage_error(labels_float, preds_float)
    print("MAPE", MAPE)

    return MAE, MAPE


def check_string_exist(pred):
    match_strings = set(["\"yes\"", "'yes'", "[\"yes\"]", "['yes']", "[yes]", "yes"])
    pred = pred.replace(",", "")
    pred = pred.replace(".", "")
    pred = pred.replace("\"", "")
    pred_words = pred.split(" ")
    pred_words = [word.lower().strip() for word in pred_words]
    pred_words = set(pred_words)

    match_results = list(match_strings.intersection(pred_words))

    if len(match_results) > 0:
        return 1
    else:
        return 0


def parse_score(answer):
    score_index = answer.find('answer_choice')
    if score_index == -1:
        return "no answer"
    val = answer[score_index+15:]
    number = re.findall(r'\d+\.\d+|\d+', val)
    if number:
        extracted_number = float(number[0])
        return extracted_number
    else:
        return "no answer"
