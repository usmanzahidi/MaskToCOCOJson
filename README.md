# Mask To COCO Json

Converts annotation mask images to COCO json

## Requirements
`python3` `numpy` `opencv-python`

## Installation

```
git clone https://github.com/usmanzahidi/MaskToCOCOJson.git
```

## Usage

```bash
usage: main.py [-i PATH] [-m PATH] [-f JSONFILE] 
-i rgb image folder path
-m annotation mask images folder
-f json output file name

```

## Example:

```bash
python main.py -i ../../Dataset/fastpick/5/train/images/ -m ../../Dataset/fastpick/5/train/annotations/ -f ../../Dataset/fastpick/5/train/annotations/annotations.json

```

## Test:

```bash
Once json file is generated, annotations can be tested by using Coco-viewer from https://github.com/trsvchn/coco-viewer

```


