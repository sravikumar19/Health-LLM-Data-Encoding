import os
import pandas as pd
import numpy as np
import wfdb
import subprocess
import shutil
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import glob
import json
import random
import csv
from random import randint
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, f1_score, confusion_matrix

hz = 500
start_time = 0
time = 10
start_length = int(start_time * hz)
sample_length = int(time * hz)
end_time = start_time + time
t = np.arange(start_time, end_time, 1 / hz)
lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
random.seed(10)

def load_ecg_data(question_type, data_path):
    data = []
    for fname in sorted(glob.glob(data_path + '/*.json')):
        with open(fname, "r") as f:
            data.extend(json.load(f))
    filtered_pos = []
    filtered_neg = []
    filtered_not_sure = []
    s = set()
    for item in data:
        s.add(item['question_type'])
        if item['question_type'] == question_type and item['answer'][0] == 'yes':
            filtered_pos.append(item)
        elif item['question_type'] == question_type and item['answer'][0] == 'no':
            filtered_neg.append(item)
        elif item['question_type'] == question_type and item['answer'][0] == 'not sure':
            filtered_not_sure.append(item)

    sampled = []
    sampled_pos = random.sample(filtered_pos, 10)
    sampled_neg = random.sample(filtered_neg, 10)
    sampled.extend(sampled_pos)
    sampled.extend(sampled_neg)

    if "single" in question_type:
        sampled_not_sure = random.sample(filtered_not_sure, 5)
        sampled.extend(sampled_not_sure)

    random.shuffle(sampled)
    return sampled

def parse_pred(pred, question_type='single'):
    match_strings_yes = set(["\"yes\"", "'yes'", "[\"yes\"]", "['yes']", "[yes]", "yes"])
    match_strings_not_sure = set(["\"sure\"", "'sure'", "[\"sure\"]", "['sure']", "[sure]", "sure"])
    match_strings_no = set(["\"no\"", "'no'", "[\"no\"]", "['no']", "[no]", "no"])
    pred = pred.replace(",", "")
    pred = pred.replace(".", "")
    pred = pred.replace("\"", "")
    pred_words = pred.split(" ")
    pred_words = [word.lower().strip() for word in pred_words]
    pred_words = set(pred_words)

    match_results_yes = list(match_strings_yes.intersection(pred_words))
    match_results_not_sure = list(match_strings_not_sure.intersection(pred_words))

    if len(match_results_yes) > 0 or 'yes' in pred:
        return 'yes'
    elif 'single' in question_type and (len(match_results_not_sure) > 0 or 'not sure' in pred):
        return 'not sure'
    else:
        return 'no'

def compute_accuracy(labels, preds):
    acc = accuracy_score(labels, preds)
    print("accuracy", acc)
    macro_f1 = f1_score(labels, preds, average = 'macro')
    print("macro f1", macro_f1)
    micro_f1 = f1_score(labels, preds, average = 'micro')
    print("micro f1", micro_f1)
    print(confusion_matrix(labels, preds, labels=["yes", "no", "not sure"]))


def parse_response(response):
    score_index = response.find('answer_choice')
    if score_index == -1:
        return "no"
    return response[score_index+15:]


def draw_ecg(ecg, lead, output_path):
    plt.rcParams["figure.figsize"] = (25, 1.5)
    plt.subplot(1, 1, 1)
    plt.plot(
        t,
        ecg[lead][start_length: start_length + sample_length],
        linewidth=2,
        color="k",
        alpha=1.0,
        label=lead_names[lead]
    )
    minimum = min(ecg[lead])
    maximum = max(ecg[lead])
    ylims_candidates = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0 , 1.5, 2.0, 2.5]

    try:
        ylims = (
            max([x for x in ylims_candidates if x <= minimum]),
            min([x for x in ylims_candidates if x >= maximum]),
        )
    except:
        ylims = (-2.5, 2.5)
    plt.vlines(np.arange(start_time, end_time, 0.2), ylims[0], ylims[1], colors="r", alpha=1.0)
    plt.vlines(np.arange(start_time, end_time, 0.04), ylims[0], ylims[1], colors="r", alpha=0.3)
    plt.hlines(np.arange(ylims[0], ylims[1], 0.5), start_time, end_time, colors="r", alpha=1.0)
    plt.hlines(np.arange(ylims[0], ylims[1], 0.1), start_time, end_time, colors="r", alpha=0.3)

    plt.xticks(np.arange(start_time, end_time + 1, 1.0))
    plt.margins(0.2)
    plt.title(output_path + '/ECG: ' + lead_names[lead])
    plt.savefig(output_path + '/ecg_lead_' + str(lead) + '.png', bbox_inches='tight')
    plt.close()


def draw_all_leads_ecg(ecg, output_path):
    plt.rcParams["figure.figsize"] = (25, 18)
    for lead in range(len(ecg)):
        plt.subplot(len(ecg), 1, lead+1)
        plt.plot(
            t,
            ecg[lead][start_length: start_length + sample_length],
            linewidth=2,
            color="k",
            alpha=1.0,
            label=lead_names[lead]
        )
        minimum = min(ecg[lead])
        maximum = max(ecg[lead])
        ylims_candidates = [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0 , 1.5, 2.0, 2.5]

        try:
            ylims = (
                max([x for x in ylims_candidates if x <= minimum]),
                min([x for x in ylims_candidates if x >= maximum]),
            )
        except:
            ylims = (-2.5, 2.5)
        plt.vlines(np.arange(start_time, end_time, 0.2), ylims[0], ylims[1], colors="r", alpha=1.0)
        plt.vlines(np.arange(start_time, end_time, 0.04), ylims[0], ylims[1], colors="r", alpha=0.3)
        plt.hlines(np.arange(ylims[0], ylims[1], 0.5), start_time, end_time, colors="r", alpha=1.0)
        plt.hlines(np.arange(ylims[0], ylims[1], 0.1), start_time, end_time, colors="r", alpha=0.3)

        plt.xticks(np.arange(start_time, end_time + 1, 1.0))
        plt.title('ECG lead ' + str(lead))
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def write_results_to_file(results, output_path):
    with open(output_path + '/results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "question", "answer", 'response', "prediction"])
        for result in results:
            writer.writerow(result)
