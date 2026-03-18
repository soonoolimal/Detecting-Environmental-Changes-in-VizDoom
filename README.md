### 1. Create Dataset
```bash
python3 -m data.create_dataset \
    --env_name DefendLine --exp_name base \
    --num_episodes 200 --video_save_freq 20
```

### 2. Run
```bash
# pretrain dt and train td without ae
python3 main.py \
    --env_name DefendLine --exp_name expn --ds_name expn
```

```bash
# train only td based on pretrained dt
python3 main.py \
    --env_name DefendLine --exp_name expn --ds_name expn \
    --dt_pretrained \
    --pretrained_dir model/pretrained/DefendLine_expn_20260317_120000
```