#!/bin/bash

# 定义三个不同的命令
commands=(
    "python main.py --machine machine   --used_data 'GridGraphall'   --prompt_content 'node_graph' --flag=0, --task_id='dcn_Pdp1e6', "
    "python main.py --machine machine   --used_data 'GridGraphall'   --prompt_content 'node_graph' --flag=1, --task_id='dcn_Pdpm1e8"
    "python main.py --machine machine   --used_data 'GridGraphall'   --prompt_content 'node_graph' --flag=2, --task_id='dcn_Pdpm1e7"
)

# 运行每个命令
for cmd in "${commands[@]}"
do
    echo "Running: $cmd"
    eval $cmd
done
