python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/augment_dataset.py \
  --config configs/augment_offline.yaml \
  --input-root data/raw \
  --output-root data/interim/augmented

python scripts/split_dataset.py \
  --input-root data/interim/augmented \
  --output-root data/processed \
  --train 0.70 --val 0.15 --test 0.15

python scripts/audit_dataset.py --data-root data/processed
python scripts/launch_fiftyone.py --data-root data/processed
bash scripts/train_yolo26_cls.sh data/processed yolo26n-cls.pt 224 80