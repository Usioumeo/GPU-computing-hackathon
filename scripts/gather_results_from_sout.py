import os
import subprocess
import pathlib

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SbatchMan.scripts.common import parse_results_csv, summarize_results

SbM_HOME = os.environ.get('SbM_HOME')
if not SbM_HOME:
    print('Environment not set. Exiting...')
    exit(1)

METADATA_PATH = f'{SbM_HOME}/metadata/baldo'
SOUT_PATH = f'{SbM_HOME}/sout/baldo'
# GRAPH_CATEGORIES = ['BFS_smallD', 'BFS_largeD']

summary_file = pathlib.Path(f'{METADATA_PATH}/overallTable.csv')

# if not summary_file.exists():
# Generate results summary table
p = subprocess.Popen(['utils/overallTable.sh'], cwd=SbM_HOME)
p.wait()

print(f'{summary_file=}')
results = parse_results_csv(summary_file)
summarize_results(results)