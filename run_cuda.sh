#!/usr/bin/env bash

#blocks=(128 192 256 512 1024)
#blocksize=(128 192 256 512 1024)
blocks=(512 1024)
blocksize=(512 1024)

base="/work/07460/garlands/spss_cuda/USA-road-d"
PS3="Please enter your choice: "
options=("NY" "BAY" "COL" "FLA" "NW" "NE" "CAL" "LKS" "E" "W" "CTR" "USA" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "NY")
            region="$base.$opt.gr"
            break
            ;;
        "BAY")
            region="$base.$opt.gr"
            break
            ;;
        "COL")
            region="$base.$opt.gr"
            break
            ;;
        "FLA")
            region="$base.$opt.gr"
            break
            ;;
        "NW")
            region="$base.$opt.gr"
            break
            ;;
        "NE")
            region="$base.$opt.gr"
            break
            ;;
        "CAL")
            region="$base.$opt.gr"
            break
            ;;
        "LKS")
            region="$base.$opt.gr"
            break
            ;;
        "E")
            region="$base.$opt.gr"
            break
            ;;
        "W")
            region="$base.$opt.gr"
            break
            ;;
        "CTR")
            region="$base.$opt.gr"
            break
            ;;
        "USA")
            region="$base.$opt.gr"
            break
            ;;
        "Quit")
            echo "Bye"
            exit
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

printf "\nRunning $opt\n"

for block in "${blocks[@]}"; do
    for size in "${blocksize[@]}"; do
        result=$(./build/bellman cuda $region $block $size | tail -1)
<<<<<<< HEAD
        printf "Blocks: $block\tBlocksize: $size\tTime: $result\n" | tee -a ./results/$opt-result.txt
=======
        printf "Blocks: $block\tBlocksize: $size\tTime: $result\n" | tee ./results/$opt-result.txt
>>>>>>> 5b1e83a9c5786c3b81d7c90b1f83e3e23ed4b25f
    done
done
