#!/bin/sh

TMPDIR=./foo-neva-train
rm -fr ${TMPDIR}

source ~/env/bin/activate
#
#  --trace=cuda,nvtx,cublas,cudnn \
#  --trace=cuda,nvtx,cublas,osrt,cudnn \
#nsys profile \
#  --force-overwrite=true \
#  --output=/tmp/neva-a100-thunder.nsys-rep \
#  --opengl-gpu-workload=false \
#  --stats=false \
#  --show-output=true \
#  --trace=cuda,nvtx,cublas,cudnn \
#

HYDRA_FULL_ERROR=1 \
THUNDER_ANNOTATE_TRACES=1 \
NEMO_THUNDER_NEVA=thunder \
python3 \
	./examples/multimodal/multimodal_llm/neva/neva_pretrain.py \
         trainer.precision=16 \
         model.megatron_amp_O2=False \
         trainer.num_nodes=1 \
         trainer.devices=1 \
         trainer.val_check_interval=10 \
         trainer.limit_val_batches=5 \
         trainer.log_every_n_steps=1 \
         ++exp_manager.max_time_per_run=00:00:03:00 \
         trainer.max_steps=20 \
         model.micro_batch_size=2 \
         model.global_batch_size=4 \
         model.tensor_model_parallel_size=1 \
         model.pipeline_model_parallel_size=1 \
         exp_manager.create_checkpoint_callback=False \
         model.data.data_path=./data/multimodal/tiny-neva/dummy.json \
         model.data.image_folder=./data/multimodal/tiny-neva/images \
         model.tokenizer.library=sentencepiece \
         model.tokenizer.model=./data/multimodal/tiny-neva/tokenizer_add_special.model \
         model.num_layers=2 \
         model.hidden_size=5120 \
         model.ffn_hidden_size=13824 \
         model.num_attention_heads=40 \
         model.normalization=rmsnorm \
         model.data.num_workers=0 \
         model.data.conv_template=llama_2 \
         model.mm_cfg.vision_encoder.from_pretrained='openai/clip-vit-large-patch14' \
         model.mm_cfg.llm.from_pretrained=null \
         model.use_flash_attention=false \
         exp_manager.exp_dir=${TMPDIR}

rm -fr ${TMPDIR}
#cp /tmp/neva-a100-thunder.nsys-rep ~/share/
