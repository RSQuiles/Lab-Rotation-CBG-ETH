import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help= "Directory containing genes.npy")
parser.add_argument('--shard-size', type=int, default=100_000)

args = parser.parse_args()

print("Loading genes.npy...")
genes = np.load(f"{args.dir}/genes.npy", mmap_mode='r')
num_samples = genes.shape[0]
shard_size = args.shard_size

for i in range(0, num_samples, shard_size):
    print(f"Generating shard: {i//shard_size}")
    np.save(f"{args.dir}/genes_part{i//shard_size}.npy", genes[i:i+shard_size])