import pnlp
import numpy as np
from collections import Counter


def main(in_file, out_file):
    lines = pnlp.read_lines(in_file)
    length = len(lines)
    num = int(0.1 * length)
    need = np.random.choice(lines, num, replace=False)
    labels = [v.split("\t")[1] for v in need]
    count = Counter(labels)
    print(len(count))
    print(sorted(count.items(), key=lambda x: x[0]))
    pnlp.write_file(out_file, need.tolist())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--in_file', type=str)
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()
    main(args.in_file, args.out_file)
