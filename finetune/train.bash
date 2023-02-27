# 请下载APPS数据集

python -m torch.distributed.launch --nproc_per_node=2 tune_apps_gpt.py  \
--save-dir=../models/gpt_neo_125_finetuned  \
--load=EleutherAI/gpt-neo-125M \
--apps-train-files=/home2/APPS/train \
--apps-dataroot=/home2/APPS/train \
--grad-acc-steps=8 \
--epochs=2 \
--batch-size-per-replica=8