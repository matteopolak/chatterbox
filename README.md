# Chat Generator

![Build Status](https://github.com/matteopolak/chat-generator/actions/workflows/yapf.yml/badge.svg)
[![License:MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Training on new text

```bash
python src/train.py --model atrioc --data data/atrioc.csv
```

### Chat dataset format

The message dataset must be a CSV file where the first column is the username of the chatter, and the second column is the message content. The column header should not be present. For example:

```csv
mattepolak,"Hello, world!"
```

### Using a GPU

To use a GPU, [follow the official instructions](https://www.tensorflow.org/install/pip#step-by-step_instructions).

## Generating text

```bash
python src/generate.py --model atrioc --length 500 --text "atrioc: i love"
```
