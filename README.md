### 1. Create Dataset
```bash
python3 -m data.create_dataset \
    --env DefendLine --exp default \
    --num_episodes 100 --video_save_freq 20
```

### 2. Run
```bash
# pretrain dt and train td without ae
python3 main.py \
    --env DefendLine --exp default
```

```bash
# train only td based on pretrained dt
python3 main.py \
    --env DefendLine --exp default \
    --dt_pretrained
```