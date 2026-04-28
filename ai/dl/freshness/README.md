# Freshness DL

Harvest Slot freshness/quality grading MVP implementation.

## Dataset Structure

```text
ai/dl/freshness/data/
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

Use `apple` first. Later, add the same structure for `pear`, `citrus`, and other fruit types.

## Sample QC Data

The QC manual says this dataset is organized by agricultural product quality grades:

```text
special / high / normal
```

For the current apple Fuji sample, use this project mapping:

```text
apple_fuji_L -> A -> special grade
apple_fuji_M -> B -> high grade
apple_fuji_S -> C -> normal grade
```

Prepare the sample images for ImageFolder training:

```powershell
python ai/dl/freshness/prepare_sample_data.py --overwrite
```

This copies images from `Data/Sample/원천데이터/1.Apple/apple_fuji` into `ai/dl/freshness/data/train|valid|test/apple/A|B|C`.

The QC guideline recommends this split:

```text
train 70% / validation 10% / test 20%
```

## Train

```powershell
python ai/dl/freshness/train.py --fruit-type apple --epochs 10 --batch-size 16
```

Use pretrained weights only when they are already available locally or network access is allowed:

```powershell
python ai/dl/freshness/train.py --fruit-type apple --pretrained
```

## Infer

```powershell
python ai/dl/freshness/infer.py --image path/to/apple.jpg --checkpoint ai/dl/freshness/models/apple_resnet18_best.pt
```

The inference result matches the planned API response fields: grade, freshness score, color score, roundness score, bruise probability, shipping decision, and model confidence.
