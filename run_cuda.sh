#!/usr/bin/env bash

blocks=(128 160 192 224 256 288 320 352 384 416 448 480 512 640 768 896 1024)
blocksize=(128 160 192 224 256 288 320 352 384 416 448 480 512 640 768 896 1024)

for block in "${blocks[@]}"; do
    for size in "${blocksize[@]}"; do
        result=$(./build/bellman cuda ./input/USA-road-d.NY.gr $block $size | tail -1)
        printf "Blocks: $block\tBlocksize: $size\tTime: $result\n" >> ./result.txt
    done
done