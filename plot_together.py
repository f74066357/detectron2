import json
import matplotlib.pyplot as plt


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


experiment_folder = './output2/'
experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

plt.figure()
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
plt.legend(['total_loss'], loc='upper left')
plt.savefig("output2/loss.png")

plt.figure()
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'segm/AP' in x],
    [x['segm/AP50'] for x in experiment_metrics if 'segm/AP' in x])
plt.legend(['segm/AP'], loc='upper left')
plt.savefig("output2/acc.png")
