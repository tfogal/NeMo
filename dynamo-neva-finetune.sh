#!/bin/sh

# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

TMPDIR=./foo-neva-train
# profile="/tmp/neva-a100-dyn-thunder.nsys-rep"
rm -fr ${TMPDIR}

#source ~/env/bin/activate
#
#  --trace=cuda,nvtx,cublas,cudnn \
#  --trace=cuda,nvtx,cublas,osrt,cudnn \
#nsys profile \
#  --force-overwrite=true \
#  --output=${profile} \
#  --opengl-gpu-workload=false \
#  --stats=false \
#  --show-output=true \
#  --trace=cuda,nvtx,cublas,cudnn \
#
# nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o dynamo_thunder_neva \

# rm -f /tmp/graph*.log.txt


# /scratch/nsight-systems-linux-public-DVS/target-linux-x64/nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o dynamo_thunder_neva_finetune \

HYDRA_FULL_ERROR=1 \
THUNDER_ANNOTATE_TRACES=1 \
NEMO_THUNDER_NEVA=dynamo \
python3 \
	./examples/multimodal/multimodal_llm/neva/neva_finetune.py \
         --config-path=/scratch/thunder_neva/tfogal_nemo_neva \
         --config-name=llama3_8b_chat.yaml \
         exp_manager.create_wandb_logger=False \
         trainer.precision=bf16-mixed \
         model.megatron_amp_O2=True \
         model.mcore_gpt=False \
         trainer.num_nodes=1 \
         trainer.devices=1 \
         trainer.val_check_interval=100 \
         trainer.limit_val_batches=5 \
         trainer.log_every_n_steps=1 \
         ++exp_manager.max_time_per_run=00:00:03:00 \
         trainer.max_steps=50 \
         model.optim.lr=1e-5 \
         model.micro_batch_size=12 \
         model.global_batch_size=12 \
         model.tensor_model_parallel_size=1 \
         model.pipeline_model_parallel_size=1 \
         exp_manager.create_checkpoint_callback=False \
         model.data.data_path=./data/multimodal/tiny-neva/dummy.json \
         model.data.image_folder=./data/multimodal/tiny-neva/images \
         model.tokenizer.library=sentencepiece \
         model.tokenizer.model=./data/multimodal/tiny-neva/tokenizer_add_special.model \
         model.num_layers=16 \
         model.hidden_size=4096 \
         model.ffn_hidden_size=14336 \
         model.num_attention_heads=32 \
         model.normalization=rmsnorm \
         model.data.num_workers=0 \
         model.mm_cfg.llm.model_type=llama_2 \
         model.data.conv_template=llama_2 \
         model.mm_cfg.vision_encoder.from_pretrained='openai/clip-vit-large-patch14-336' \
         model.mm_cfg.llm.from_pretrained=null \
         model.use_flash_attention=false \
         ++model.mm_cfg.vision_encoder.crop_size=[336,336] \
         +model.data.image_token_len=576 \
         exp_manager.resume_if_exists=False \
         exp_manager.create_checkpoint_callback=False \
         model.data.image_aspect_ratio=pad \
         exp_manager.exp_dir=${TMPDIR} \
         exp_manager.explicit_log_dir=${TMPDIR} \
         model.nsys_profile.enabled=True \
         model.nsys_profile.start_step=3 \
         model.nsys_profile.end_step=4 \
         model.nsys_profile.gen_shape=True \
         exp_manager.wandb_logger_kwargs.name=neva_llama3_8b_chat_fine_tuning
         
         


rm -fr ${TMPDIR}
if test -f ${profile} ; then
  cp ${profile} ~/share/
fi
