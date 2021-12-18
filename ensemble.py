# Import base libraries
import sys
import glob, os

from tqdm import tqdm
import argparse

from clean import clean_contents

# Execute main functionality
if __name__ == '__main__':
    def to_list(arg):
        return [str(i) for i in arg.split(",")]

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument("-f", "--folder", help="The folder path containing the log file(s)", type=str, nargs="?")
    parser.add_argument("-l", "--logs", help="The path to the log file(s)", type=to_list, nargs="?")
    args = parser.parse_args()

    log_files = []
    if args.logs is not None:
        log_files = args.logs
    elif args.folder is not None:
        for file in os.listdir(args.folder):
            if file.endswith(".log"):
                log_files.append(args.folder + "/" + file)

    for path in log_files:
        with open(path) as f:
            contents = f.readlines()
            head, body = clean_contents(contents)
            print("#" + path)
            print(head) 
            print(body)