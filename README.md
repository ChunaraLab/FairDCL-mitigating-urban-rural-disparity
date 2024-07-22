# FairDCL-mitigating-urban-rural-disparity

Codes for "Mitigating urban-rural disparities in contrastive representation learning with satellite imagery", 2024 AAAI/ACM Conference on AI, Ethics, and Society.

Supplementary material of the paper: https://drive.google.com/file/d/1smhF0G11SBmlLCvXZnQ5pBAe5l7N00wb/view?usp=sharing


Start the training with:

```
python3 main_moco.py \
-a resnet50 \
--lr 0.002 \
--batch-size 32 \
--moco-dim 512 \
--mlp --moco-t 0.2 --aug-plus --cos \
  <Directory of training data>
```
