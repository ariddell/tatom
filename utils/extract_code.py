#! /usr/bin/env python
"""Extract code from restructured text files"""

from io import StringIO
import logging
import os
import pickle
import re
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO)

#----------------------------------------------------
# Globals
#----------------------------------------------------
file_path = os.path.dirname(__file__)
docs_rst_dir = os.path.join(file_path, 'source')
code_dir = os.path.join(file_path, 'source', 'generated')

#----------------------------------------------------
# Extract code from restructuredtext files
#----------------------------------------------------

def extract_code(filename, output_filename):
    if not filename.endswith('.rst'):
        raise ValueError('rst file expected.')
    output = []
    successive_lines = 0
    with open(filename) as f:
        for line in f:
            if line.startswith(".. "):
                in_python_block = True if "python" in line else False
            # exclude ":suppress:" and such
            if line.startswith(" " * 4) and in_python_block:
                successive_lines += 1
                line = re.sub(r'^    ', '', line)
                if not line[0] in ":!%@":
                    output.append(line)
            else:
                if line == '':
                    continue
                if successive_lines > 2:
                    output.extend(['\n\n', '##', '\n\n'])
                successive_lines = 0

    with open(output_filename, 'w') as f:
        logging.info('Wrote to {}'.format(output_filename))
        f.write(''.join(output))
    logging.info("extracted code from {}".format(filename))


if __name__ == "__main__":
    fn = sys.argv[1]
    fn_out = os.path.basename(fn).replace('.rst', '.py')
    extract_code(fn, fn_out)
    mod_time = os.stat(fn).st_mtime
    logging.info("monitoring for changes.")
    try:
        while True:
            time.sleep(1)
            new_mod_time = os.stat(fn).st_mtime
            if new_mod_time > mod_time:
                logging.info("Detected change, reloading.")
                extract_code(fn, fn_out)
                mod_time = new_mod_time
    except KeyboardInterrupt:
        logging.info("caught interrupt, exiting...")
        sys.exit(1)
