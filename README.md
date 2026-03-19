## 1. Create Dataset
```bash
python3 -m data.create_dataset \
    --env DefendLine --exp base --seed 0 \
    --shift all --robj all \
    --timeout 2100 --frameskip 3 --n 300 --vsf 30
```