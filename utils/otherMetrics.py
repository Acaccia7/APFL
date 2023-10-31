import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_evaluation_metrics(y_true, y_pred):
    evaluation_metrics = {
        'Precision': [],
        'FPR': [],
        'FNR': [],
        'F1': []
    }
    for i in range(6):
        y_true_i = [int(label == i) for label in y_true]
        y_pred_i = [int(label == i) for label in y_pred]
        # precision = precision_score(y_true_i, y_pred_i)
        precision = recall_score(y_true_i, y_pred_i, pos_label=1)
        fpr = 1 - recall_score(y_true_i, y_pred_i, pos_label=0)
        fnr = 1 - recall_score(y_true_i, y_pred_i, pos_label=1)
        f1 = f1_score(y_true_i, y_pred_i)
        evaluation_metrics['Precision'].append(precision)
        evaluation_metrics['FPR'].append(fpr)
        evaluation_metrics['FNR'].append(fnr)
        evaluation_metrics['F1'].append(f1)
    return evaluation_metrics


def plot_evaluation_heatmap(evaluation_metrics, filename):
    attack_types = ["Benign", "Botnet ARES", "Brute Force", "DoS/DDoS", "PortScan", "Web Attack"]
    metrics = ["Precision", "FPR", "FNR", "F1"]
    values = [evaluation_metrics[m] for m in metrics]
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap='YlGn')
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticks(np.arange(len(attack_types)))
    ax.set_yticklabels(metrics)
    ax.set_xticklabels(attack_types)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(metrics)):
        for j in range(len(attack_types)):
            text = ax.text(j, i, round(values[i][j], 2), ha='center', va='center', color='black')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va='bottom')
    fig.tight_layout()
    fig.savefig('../images' + filename, dpi=300)
    plt.show()


def plot_evaluation_table(evaluation_metrics):
    attack_types = ["Benign", "Botnet ARES", "Brute Force", "DoS/DDoS", "PortScan", "Web Attack"]
    metrics = ["Precision", "FPR", "FNR", "F1"]
    data = {attack_type: [f"{evaluation_metrics[m][i]:.3f}" for m in metrics] for i, attack_type in
            enumerate(attack_types)}
    df = pd.DataFrame(data=data, index=metrics)
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
    table = ax.tables[0]
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    fig.tight_layout()
    # plt.show()

