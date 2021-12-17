# Import base libraries
import sys

from tqdm import tqdm
import argparse

from clean import clean_contents

# Execute main functionality
if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("log_1", help="The path to the log file", type=str)
    parser.add_argument("log_2", help="The path to the log file", type=str)
    parser.add_argument("log_3", help="The path to the log file", type=str)
    args = parser.parse_args()

    for path in [args.log_1, args.log_2, args.log_3]:
        with open(path) as f:
            contents = f.readlines()
            head, body = clean_contents(contents)
            print(head) 
            print(body)