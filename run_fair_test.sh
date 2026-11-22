#!/bin/bash
# Fair test on Colab: your prompts vs. a BROAD mixed set (all topics, coding included),
# plus a cross-check with another person's prompts.
set -e
MODEL=${MODEL:-allenai/OLMoE-1B-7B-0125-Instruct}
PERSONAL=${PERSONAL:-prompts/personal_sample.jsonl}
OUT=${OUT:-results_olmoe}
ARGS="--skip-template --max-new-tokens 32 --gpu-mem 13GiB"
mkdir -p $OUT
[ -f $OUT/personal.npz ] || python profile_experts.py --model $MODEL --prompts $PERSONAL --out $OUT/personal.npz $ARGS
python profile_experts.py --model $MODEL --prompts prompts/generic_mixed.jsonl --out $OUT/generic_mixed.npz $ARGS
python profile_experts.py --model $MODEL --prompts prompts/other_person.jsonl --out $OUT/other_person.npz $ARGS
python analyze.py --personal $OUT/personal.npz --generic $OUT/generic_mixed.npz --cross $OUT/other_person.npz --outdir $OUT/fair
