# Download data 
put [multiome](https://www.dropbox.com/sh/70caiyjydx3jnq1/AAB51h6PCX9IGgi8jyT5KMhaa?dl=0) data folder under data/  
- dataset:
  - Brain
    - train: AD
    - test: human_brain_3k
  - Fetal
    - train: fetal
    - test: 
# Run 
python train_clip.py --data_dir AD --logit_scale 1

