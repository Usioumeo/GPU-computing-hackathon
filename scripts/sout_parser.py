import re
from typing import Union

LINE_PREFIXES_WHITELIST = ['[OUT]']

# RE_IGNORE_OPTIONAL_LINE = r'(?:[^\n]*)?'
# RE_IGNORE_LINES = r'(?:[^\n]*)*'
PATTERN = [
    r'\[OUT\] -- BFS iteration #\d+, source=\d+ --',
    r'\[OUT\] Total BASELINE BFS time: (\d+\.\d+) ms',
    r'\[OUT\] Graph diameter: (\d+)',
    r'\[OUT\] Total BFS time: (\d+\.\d+) ms',
]
PATTERN = r'\n'.join(PATTERN)
# print(f'Pattern: {PATTERN}')

def parse_stdout_file(content: str) -> Union[None,list[tuple[float, int, float]]]:
    if '[OUT] ALL RESULTS ARE CORRECT' not in content:
        return None
    
    # Remove colors
    # content = re.subn(r'\^\[\[\d+m', '', content)[0]

    # Keep only relevant lines
    lines = content.split('\n')
    lines_ok = []
    for line in lines:
        for pfx in LINE_PREFIXES_WHITELIST:
            if line.startswith(pfx):
                lines_ok.append(line)
                break
            
    content = '\n'.join(lines_ok)

    return re.findall(PATTERN, content)
