# SimpleQA Task

This task evaluates language models on their ability to answer short, fact-seeking questions. The dataset is sourced from [SimpleQA](https://huggingface.co/datasets/basicv8vc/SimpleQA) on HuggingFace.

## Dataset Information

- **Source**: [SimpleQA](https://huggingface.co/datasets/basicv8vc/SimpleQA)
- **Size**: 4,326 examples
- **Format**: Each example contains:
  - `problem`: The question to be answered
  - `answer`: The ground truth answer
  - `metadata`: Additional information including topic, answer type, and source URLs

## Task Format

The task presents questions in a simple format:
```
Question: <question>
Answer:
```

The model is expected to generate the answer that matches the ground truth answer.

## Evaluation

The task uses exact match as the evaluation metric, with case and punctuation insensitivity.

## Usage

To evaluate a model on this task:

```bash
python main.py --model hf --model_args pretrained=<model_name> --tasks simpleqa --limit 1
```

```bash
python -m scripts.write_out     --output_base_path output/simpleqa --tasks simpleqa    --num_examples 10
```