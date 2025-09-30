torchrun --nproc-per-node=1 --master_port=61107 sampleGPCBD.py \
    --model_path "ckpt/final.bin" \
    --steps 40000 \
    --input_path input_pf \
    --output_path output \
    --repeat_num 4 \
    --uid_list "" \
    --temperature 0.5 