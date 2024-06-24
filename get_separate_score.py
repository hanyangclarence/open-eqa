import json
import numpy as np

gt_path = 'data/open-eqa-subset-questions.json'
pred_path = 'data/metrics/open-eqa-subset-questionsbaseline-gpt-4-vision-preview-1234-metrics.json'

separate_scores = {}
gt = json.load(open(gt_path))
pred = json.load(open(pred_path))
for question_id, score in pred.items():
    question = [q for q in gt if q['question_id'] == question_id][0]
    category = question['category']
    if category not in separate_scores:
        separate_scores[category] = []
    separate_scores[category].append(score)

total_scores = []
for category, scores in separate_scores.items():
    total_scores.extend(scores)
    scores = np.array(scores)
    scores = 100.0 * (scores - 1.0) / 4.0
    scores = np.mean(scores)
    print(f'{category}: {scores:.2f}')

total_scores = np.array(total_scores)
total_scores = 100.0 * (total_scores - 1.0) / 4.0
total_scores = np.mean(total_scores)
print(f'Total: {total_scores:.2f}')
