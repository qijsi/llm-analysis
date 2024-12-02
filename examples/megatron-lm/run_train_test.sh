#!/bin/bash
# Copyright 2023 chengli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

output_dir="outputs_train"
gpu_name="a100-sxm-80gb"
total_num_tokens=270e9
global_batch_size=128

if [[ ! -e $output_dir ]]; then
    mkdir $output_dir
elif [[ ! -d $output_dir ]]; then
    echo "$output_dir already exists but is not a directory" 1>&2
fi

python -m llm_analysis.analysis train --model_name megatron-lm-175b --total_num_tokens ${total_num_tokens} --gpu_name ${gpu_name} --activation_recomputation 0 --tp_size 8 --pp_size 1 --ds_zero 0 --batch-size-per-gpu 8 --global_batch_size ${global_batch_size} --output_dir ${output_dir} --total_num_gpus 8 --seq-len 512
