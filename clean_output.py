import sys
import re

def clean_contents(contents):
    id = '\[[a-zA-Z0-9]+\]'
    output = ""

    for content in contents:
        stripped = re.sub(id, '', content)
        prefix = '^   [mean_|hits_].*'

        if len(re.findall(prefix, stripped)) > 0:
            result = re.findall(prefix, stripped)
            output = output + result[0] + "\n"

    head = "label, value"
    body = output.replace(':',",").replace("   ", "")
    return head, body

if __name__ == '__main__':
    try:
        _, INPUT = sys.argv
    except Exception as e:
        print('Usage: python clean.py INPUT')
        sys.exit(0)

    filename = INPUT
    with open(INPUT) as f:
        contents = f.readlines()
        head, body = clean_contents(contents)
        print(head) 
        print(body)