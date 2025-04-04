import datasets

def process_docs(dataset: datasets.Dataset):
    """Process the SimpleQA dataset."""
    def _process_doc(doc):
        return {
            "question": doc["problem"],
            "answer": doc["answer"],
            # "metadata": doc["metadata"]
        }
    return dataset.map(_process_doc) 