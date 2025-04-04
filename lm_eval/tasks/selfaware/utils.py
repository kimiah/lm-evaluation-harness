import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        return {
            "question": doc["question"],
            "answer": doc["answer"] if doc["answerable"] else [" "],
            "answerable": doc["answerable"]
        }
    
    return dataset.map(_process_doc)
