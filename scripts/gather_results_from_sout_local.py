import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SbatchMan.scripts.common import Experiment, STATUS

SOUT_PATH = 'results'

graphs = {
    "wikipedia-20070206": {"N": 3_000_000, "M": 45_000_000, "Category": "Small-diameter"},
    "soc-LiveJournal1": {"N": 5_000_000, "M": 69_000_000, "Category": "Small-diameter"},
    "hollywood-2009": {"N": 1_000_000, "M": 114_000_000, "Category": "Small-diameter"},
    "roadNet-CA": {"N": 1_000_000, "M": 5_000_000, "Category": "Small-diameter"},
    # "GAP-twitter": {"N": 61_000_000, "M": 1_500_000_000, "Category": "Small-diameter"},
    # "GAP-web": {"N": 50_000_000, "M": 1_900_000_000, "Category": "Small-diameter"},

    "GAP-road": {"N": 23_000_000, "M": 58_000_000, "Category": "Large-diameter"},
    "rgg_n_2_22_s0": {"N": 4_000_000, "M": 60_000_000, "Category": "Large-diameter"},
    "europe_osm": {"N": 50_000_000, "M": 108_000_000, "Category": "Large-diameter"},
    "rgg_n_2_24_s0": {"N": 18_000_000, "M": 265_000_000, "Category": "Large-diameter"},

    "graph500_20_8": {"N": 1_000_000, "M": 8_000_000, "Category": "Graph500"},
    "graph500_21_8": {"N": 2_000_000, "M": 17_000_000, "Category": "Graph500"},
    "graph500_20_32": {"N": 1_000_000, "M": 33_000_000, "Category": "Graph500"},
    "graph500_21_16": {"N": 2_000_000, "M": 33_000_000, "Category": "Graph500"},
}

results = {}

# Read .out files
for name, graph in graphs.items():
    input_filename = f'results/res_{name}.out'
    if not os.path.exists(input_filename):
        print(f"No available results for graph: {name} (file: {input_filename})")
        continue
    with open(input_filename, 'r') as input_file:
        out = input_file.read()

        if graph['Category'] not in results:
            results[graph['Category']] = []

        results[graph['Category']].append((Experiment(0, name, {}, STATUS.OK if out.endswith('Return code: 0\n') else STATUS.ERROR), out))

for category, category_res in results.items():
    for exp, out in category_res:
        if exp.status != STATUS.OK:
            print('Experiment failed: ', end='')
            print(exp)
            print(out)
            continue
        else:
            print(f'Experiment on graph {exp.name:<20}: OK')

        # Do whatever you want here. out is the stdout+stderr of bin/bfs       