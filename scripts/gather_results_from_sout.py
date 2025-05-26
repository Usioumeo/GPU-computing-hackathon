import datetime
import json
import os
import subprocess
import pathlib
import sys
from statistics import geometric_mean, mean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SbatchMan.scripts.common import STATUS, parse_results_csv, summarize_results

from sout_parser import parse_stdout_file

submission_flag = '--submission' in sys.argv

SbM_HOME = os.environ.get('SbM_HOME')
if not SbM_HOME:
    print('SbatchMan Environment not set. Exiting...')
    exit(1)

GROUP_NAME = os.environ.get('GROUP_NAME')
if not GROUP_NAME:
    print('Environment not set. Exiting...')
    exit(1)

HOST = os.environ.get('HOST', 'baldo')
METADATA_PATH = f'{SbM_HOME}/metadata/{HOST}'
SOUT_PATH = f'{SbM_HOME}/sout/{HOST}'

summary_file = pathlib.Path(f'{METADATA_PATH}/overallTable.csv')

# if not summary_file.exists():
# Generate results summary table
p = subprocess.Popen(['utils/overallTable.sh'], cwd=SbM_HOME, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
p.wait()

# print(f'{summary_file=}')
results = parse_results_csv(summary_file)
if not submission_flag:
    print(summarize_results(results))

# Read STDOUT files
output = {
    'group': GROUP_NAME,
    'timestamp': datetime.datetime.today().strftime('%Y%m%d%H%M%S'),
    'geomean': 0,
    'geomeans': {},
    'speedups': {},
}
runtimes = []
for category, category_res in results.items():
    category_runtimes = []
    for exp in category_res:
        if exp.status != STATUS.OK and not submission_flag:
            print('Experiment failed: ', end='')
            print(exp)
            continue

        sout_filename = f'{SOUT_PATH}/{category}/{category}_{exp.id}.out'
        # print(f'{sout_filename=}')
        with open(sout_filename, 'r') as sout_file:
            graph_name = exp.params['-f']
            if '.' in graph_name:
                graph_name = graph_name.split('.')[0]
            sout = sout_file.read()
            # print('\n\n\n'+graph_name)
            # print(sout)
            times = parse_stdout_file(sout)
            # print(f'{times=}')
            if times:
                speedups = []
                for t in times:
                    diameter = float(t[1])
                    if diameter > 6:
                        time_baseline = float(t[0])
                        time_bfs = float(t[2])
                        speedups.append(time_baseline / time_bfs)
                speedups_avg = mean(speedups)
                category_runtimes.append(speedups_avg)
                runtimes.append(speedups_avg)
                output['speedups'][graph_name] = speedups_avg

    output['geomeans'][category] = geometric_mean(category_runtimes) if len(category_runtimes) > 0 else 0
    # print(f'[{category}] Geomean: {geometric_mean(category_runtimes)} ms')

output['geomean'] = geometric_mean(runtimes) if len(runtimes) > 0 else 0
# print(f'[OVERALL] Geomean: {geometric_mean(runtimes)} ms')

if submission_flag:
    print(json.dumps(output))
else:
    print(json.dumps(output, indent=2))
