# Freshness DL Legacy Scripts

This folder keeps the earlier Python implementation in a simpler location.
The Kaggle notebook is now the main training pipeline, so these scripts are only for local experiments or future FastAPI extraction.

## Current Workspace Layout

```text
smart_dl/
  Note/
    DL_freshness_training_pipeline.ipynb
  Docs/
  Data/
    Sample/
  models/
    apple_resnet18_best.pt
  src/
    freshness/
```

## Dataset Structure

If you use these scripts locally, they expect ImageFolder-style data:

```text
src/freshness/data/
  train/apple/A
  train/apple/B
  train/apple/C
  valid/apple/A
  valid/apple/B
  valid/apple/C
  test/apple/A
  test/apple/B
  test/apple/C
```

Project fruit codes are `apple`, `pear`, `mandarine`, and `persimmon`.

## Prepare Sample Data

```powershell
python src/freshness/prepare_sample_data.py --overwrite
```

This copies the local apple sample from `Data/Sample` into `src/freshness/data/train|valid|test/apple/A|B|C`.

## Train

```powershell
python src/freshness/train.py --fruit-type apple --epochs 10 --batch-size 16
```

Use pretrained weights only when they are already available locally or network access is allowed:

```powershell
python src/freshness/train.py --fruit-type apple --pretrained
```

## Infer

```powershell
python src/freshness/infer.py --image path/to/apple.jpg --checkpoint models/apple_resnet18_best.pt
```

The notebook stores final checkpoints as `models/{fruit}_resnet18_best.pt` locally and `/kaggle/working/models/{fruit}_resnet18_best.pt` on Kaggle.
