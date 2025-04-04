# AmbigQA Task

This task evaluates models on the AmbigQA dataset, which contains ambiguous open-domain questions that require multiple interpretations and answers.

## Dataset Information

- **Source**: [AmbigQA Dataset](https://huggingface.co/datasets/sewon/ambig_qa)
- **Paper**: [AmbigQA: Answering Ambiguous Open-domain Questions](https://arxiv.org/abs/2004.10645)
- **Size**: 14,042 questions
- **Splits**: 
  - Train: 10,036 examples
  - Validation: 2,002 examples

## Task Variants

This task includes three variants:

1. **ambigqa_all**: Includes all questions from the dataset
   - Task name: `ambigqa_all`
   - Includes both ambiguous and non-ambiguous questions
   - Uses `process_docs` for processing

2. **ambigqa_ambiguous**: Only includes ambiguous questions
   - Task name: `ambigqa_ambiguous`
   - Filters for questions with multiple interpretations
   - Uses `process_docs_ambiguous` for processing

3. **ambigqa_nonambiguous**: Only includes non-ambiguous questions
   - Task name: `ambigqa_nonambiguous`
   - Filters for questions with single interpretations
   - Uses `process_docs_nonambiguous` for processing

## Task Format

The task presents questions to the model and expects it to provide answers. Questions can have either:
- Single answers (non-ambiguous)
- Multiple interpretations requiring multiple answers (ambiguous)

Multiple answers are joined with semicolons in the target format.

Each processed document includes:
- The question
- The answer(s)
- A boolean flag indicating if the question is ambiguous
