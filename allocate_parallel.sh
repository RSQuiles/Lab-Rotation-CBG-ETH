#!/bin/bash
salloc \
  --job-name=parallel_debug \
  --ntasks=2 \
  --mem-per-cpu=32G \
  --cpus-per-task=4 \
  --gpus=2 \
  --time=2:00:00 \
  bash
