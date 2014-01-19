#!/usr/bin/env python3
"""Split a single text into chunks and save chunks"""

import argparse
import os
import string
import logging

logging.basicConfig(level=logging.INFO)


def split_text(filename, num_words):
    """Split a long text into chunks of approximately `num_words` words."""
    with open(filename, 'r') as input:
        words = input.read().split(' ')
    chunks = []
    current_chunk_words = []
    current_chunk_word_count = 0
    for word in words:
        current_chunk_words.append(word)
        if word not in string.whitespace:
            current_chunk_word_count += 1
        if current_chunk_word_count == num_words:
            chunk = ' '.join(current_chunk_words)
            chunks.append(chunk)
            # start over for the next chunk
            current_chunk_words = []
            current_chunk_word_count = 0
    final_chunk = ' '.join(current_chunk_words)
    chunks.append(final_chunk)
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split text')
    parser.add_argument('num_words', type=int, help='Chunk word length')
    parser.add_argument('input', type=str, help='Input text')
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    filename = args.input
    output_dir = args.output_dir
    chunk_length_words = args.num_words
    chunks = split_text(filename, chunk_length_words)
    # we want a suffix that identifies the chunk, such as "02" for the 2nd
    # chunk. Python has a couple of standard ways of doing this that should
    # be familiar from other programming languages:
    # "%04d" % 2 => "0002"
    # "{:04d}".format(2) => "0002"
    fn = os.path.basename(filename)
    fn_base, fn_ext = os.path.splitext(fn)
    for i, chunk in enumerate(chunks):
        chunk_filename = "{}{:04d}{}".format(fn_base, i, fn_ext)
        with open(os.path.join(output_dir, chunk_filename), 'w') as f:
            f.write(chunk)
    logging.info("Split {} into {} files. Saved to {}".format(filename, len(chunks), output_dir))
