from math import ceil
import os
import json
from operator import itemgetter
import sys
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib

FONT_TITLE = 18
FONT_AXES = 18
FONT_TICKS = 16
FONT_LEGEND = 14

plt.rc('axes', titlesize=FONT_AXES)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_AXES)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_TICKS)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_TICKS)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_LEGEND)    # legend fontsize
plt.rc('figure', titlesize=FONT_TITLE)  # fontsize of the figure title
# matplotlib.rcParams['font.family'] = 'Noto Color Emoji'

CATEGORIES_LABELS_DICT = {
    'BFS_largeD': 'Large-diameter',
    'BFS_smallD': 'Small-diameter',
}

# SHARED_DIR = os.environ.get('SHARED_DIR')
# if not SHARED_DIR:
#     print('Environment not set. Exiting...')
#     exit(1)

def plot_ranking(ax: matplotlib.axes.Axes, x, y, title, y_label=''):
    colors = ['gold', 'silver', 'goldenrod'] + ['cornflowerblue'] * (len(x) - 3)
    bars = ax.bar(x, y, color=colors)
    ax.set_title(title)
    if y_label: ax.set_ylabel(y_label)
    ax.grid(True, axis='y')
    ax.set_xticks(range(len(x)), x, rotation=45)

    # Add medal emojis to the first three bars
    # for i, bar in enumerate(bars):
    #     if i == 0:
    #         medal = '🥇'
    #     elif i == 1:
    #         medal = '🥈'
    #     elif i == 2:
    #         medal = '🥉'
    #     else:
    #         continue
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height(),
    #         medal,
    #         ha='center',
    #         va='bottom',
    #         fontsize=14
    #     )

# f'{SHARED_DIR}/gpu-computing-hackathon-results.json'
if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <results-json-file>')
    exit(1)

with open(sys.argv[1], 'r') as sout_file:
    input = sout_file.read()
    # Old for jsonl -- data = [json.loads(line) for line in input.strip().split('\n')]
    data = json.loads(input)

    # Filter data to keep only the submission with the highest geomean for each group
    grouped_data = {}
    for entry in data:
        group = entry['group']
        if group not in grouped_data or entry['geomean'] > grouped_data[group]['geomean']:
            grouped_data[group] = entry
    data = list(grouped_data.values())

    # Rankings
    ## Global
    ranking_global = sorted(data, key=itemgetter('geomean'), reverse=True)
    ## by Graph type
    ranking_by_type = {}
    for entry in data:
        for category, value in entry['geomeans'].items():
            if category not in ranking_by_type:
                ranking_by_type[category] = []
            ranking_by_type[category].append({'group': entry['group'], 'geomean': value})
    for category in ranking_by_type:
        ranking_by_type[category] = sorted(ranking_by_type[category], key=itemgetter('geomean'), reverse=True)
    ## by Graph
    ranking_by_graph = {}
    for entry in data:
        for dataset, value in entry['speedups'].items():
            if dataset not in ranking_by_graph:
                ranking_by_graph[dataset] = []
            ranking_by_graph[dataset].append({'group': entry['group'], 'speedup': value})
    for dataset in ranking_by_graph:
        ranking_by_graph[dataset] = sorted(ranking_by_graph[dataset], key=itemgetter('speedup'), reverse=True)

    # print(ranking_global)
    # print()
    # print(ranking_by_type)
    # print()
    # print(ranking_by_graph)

    # Plotting
    fig, axes = plt.subplots(1 + ceil(len(ranking_by_graph.keys())/3), 3, figsize=(17, 10))
    axes = axes.flatten()

    # Global ranking plot
    plot_ranking(
        axes[0],
        [entry['group'] for entry in ranking_global],
        [entry['geomean'] for entry in ranking_global],
        'Global Ranking',
        'Speedups Geomean',
    )

    # Ranking by category plots
    ax_i = 1
    for category, ranking in ranking_by_type.items():
        plot_ranking(
            axes[ax_i],
            [entry['group'] for entry in ranking],
            [entry['geomean'] for entry in ranking],
            f'Ranking for {CATEGORIES_LABELS_DICT.get(category, category)}',
        )
        ax_i += 1

    # Ranking by dataset plots
    for dataset, ranking in ranking_by_graph.items():
        plot_ranking(
            axes[ax_i],
            [entry['group'] for entry in ranking],
            [entry['speedup'] for entry in ranking],
            f'Ranking for graph "{dataset[:-4]}"',
            'Speedup' if ax_i == 3 else '',
        )
        ax_i += 1

    plt.tight_layout()
    output_path = 'ranking_plot.png'
    plt.savefig(output_path)
    print(f'Ranking plot saved to {output_path}')