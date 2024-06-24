import json
import numpy as np
import pickle

num_questions = 91

gt_path = 'data/open-eqa-subset-questions.json'
pred_path = 'data/metrics/open-eqa-subset-questions-baseline-gpt-4o-1234-metrics.json'
gt_path_length_path = 'data/gt_path_length.json'
path_length_path = 'data/qid_to_gt_pl.pkl'

with open(gt_path_length_path, 'rb') as f:
    gt_path_length_map = json.load(f)
with open(path_length_path, 'rb') as f:
    path_length_map = pickle.load(f)

def spl(path_length, gt_path_length):
    return gt_path_length / max(gt_path_length, path_length)

separate_spl = {}
separate_scores = {}
gt = json.load(open(gt_path))
pred = json.load(open(pred_path))
count = 0
for question_id, score in pred.items():
    question = [q for q in gt if q['question_id'] == question_id][0]
    gt_path_length = gt_path_length_map[question_id]
    path_length = path_length_map[question_id]
    category = question['category']
    if category not in separate_scores:
        separate_scores[category] = []
    if category not in separate_spl:
        separate_spl[category] = []
    separate_scores[category].append(score)
    separate_spl[category].append(spl(path_length, gt_path_length))
    # print(spl(path_length, gt_path_length))
    count += 1
    if count == num_questions:
        break

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
