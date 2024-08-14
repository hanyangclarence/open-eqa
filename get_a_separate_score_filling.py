import json
import numpy as np
import pickle
import os
import argparse

skip_scenes = [
    "00853-5cdEh9F2hJL",
]
skip_scene_ids = [scene.split("-")[1] for scene in skip_scenes]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
)
parser.add_argument(
    "--run_id",
    type=str,
    default="",
    help="run id (default: '')",
)
parser.add_argument(
    "--dataset",
    default="open-eqa-eval",
    type=str,
)
args = parser.parse_args()


data_path = args.data_path
path_length_name = "path_length_list.pkl"
path_length_path = os.path.join(data_path, path_length_name)

gt_path = f'data/{args.dataset}.json'
pred_path = f'data/metrics/{args.dataset}-prediction-{args.run_id}-gpt-4o-1234-metrics.json'
baseline_path = f'data/metrics/{args.dataset}-baseline-gpt-4o-1234-metrics.json'
gt_path_length_path = 'data/gt_path_length.json'
baseline_path_length_path = f'data/qid_to_baseline_pl_{args.dataset}.pkl'

with open(gt_path_length_path, 'rb') as f:
    gt_path_length_map = json.load(f)
with open(path_length_path, 'rb') as f:
    path_length_map = pickle.load(f)
with open(baseline_path_length_path, 'rb') as f:
    baseline_path_length_map = pickle.load(f)

def spl(path_length, gt_path_length):
    return gt_path_length / max(gt_path_length, path_length)

separate_spl = {}
separate_scores = {}
gt = json.load(open(gt_path))
pred = json.load(open(pred_path))
baseline = json.load(open(baseline_path))
for question_id, score in baseline.items():
    try:
        question = [q for q in gt if q['question_id'] == question_id][0]
        if question['episode_history'].split("-")[-1] in skip_scene_ids:
            continue
        gt_path_length = gt_path_length_map[question_id]
        if question_id not in pred.keys():
            path_length = baseline_path_length_map[question_id]
        else:
            path_length = path_length_map[question_id]
            score = pred[question_id]
        category = question['category']
        if category not in separate_scores:
            separate_scores[category] = []
        if category not in separate_spl:
            separate_spl[category] = []
        separate_scores[category].append(score)
        separate_spl[category].append(spl(path_length, gt_path_length))
        # print(spl(path_length, gt_path_length))
    except:
        print(question_id)
        continue

total_scores = []
total_spl = []
for category, scores in separate_scores.items():
    spl_coeffs = separate_spl[category]
    total_scores.extend(scores)
    scores = np.array(scores)
    spl_scores = np.array(spl_coeffs)
    scores = 100.0 * (scores - 1.0) / 4.0
    spl_scores = scores * spl_coeffs
    total_spl.extend(spl_scores)
    scores = np.mean(scores)
    spl_scores = np.mean(spl_scores)
    print(f'{category}: {scores:.2f}')
    print(f'{category} SPL: {spl_scores:.2f}')

total_scores = np.array(total_scores)
total_scores = 100.0 * (total_scores - 1.0) / 4.0
total_scores = np.mean(total_scores)
total_spl = np.array(total_spl)
total_spl = np.mean(total_spl)
print(f'Total: {total_scores:.2f}')
print(f'Total SPL: {total_spl:.2f}')
