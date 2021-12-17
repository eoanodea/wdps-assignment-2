import sys
import re

try:
    _, INPUT = sys.argv
except Exception as e:
    print('Usage: python starter-code.py INPUT')
    sys.exit(0)

filename = INPUT
file = open(filename, 'r')

with open(INPUT) as f:
    contents = f.readlines()
    id = '\[[a-zA-Z0-9]+\]'
    output = ""

    for content in contents:
        stripped = re.sub(id, '', content)
        prefix = '^   [mean_|hits_].*'
        if len(re.findall(prefix, stripped)) > 0:
            result = re.findall(prefix, stripped)
            output = output + result[0] + ",\n"

    print(output.replace(':',","))