import datasets

def _process_doc_base(doc):
    """Base function to process a document with optional filtering by annotation type."""
    # print(f"Processing document ================================= {doc['annotations']['type']}")
    # print(doc)

    answers = []
    is_ambiguous = False
    for i, annotation_type in enumerate(doc["annotations"]["type"]):
        if annotation_type == "singleAnswer":
            # For single answer, use answer[i] directly
            # qaPairs[i] is empty in this case
            answer = doc["annotations"]["answer"][i]
            if isinstance(answer, list):
                answers.extend(answer)
            else:
                answers.append(answer)
        else:  # multipleQAs
            is_ambiguous = True
            # For multiple QAs, use qaPairs[i]
            # answer[i] is empty in this case
            qa_pair = doc["annotations"]["qaPairs"][i]
            # print(f"Processing qa_pair: {qa_pair}")
            if isinstance(qa_pair, dict) and "answer" in qa_pair:
                # print(f"Found answers in qa_pair: {qa_pair['answer']}")
                # Each answer in qa_pair['answer'] is a list of strings
                for answer_list in qa_pair["answer"]:
                    if isinstance(answer_list, list):
                        answers.extend(answer_list)
                    else:
                        answers.append(answer_list)

    # Remove duplicates from answers
    seen = set()
    answers = [x for x in answers if not (x in seen or seen.add(x))]

    if not answers:
        answers = [" "]

    # print("Analysed document ================================================")
    # print(f"Question: {doc['question']}")
    # print(f"Answers: {answers}")
    # print(f"Is ambiguous: {is_ambiguous}")

    return {
        "question": doc["question"],
        "answer": answers,
        "is_ambiguous": is_ambiguous
    }

def _has_annotation_type(doc, filter_type):
    """Check if document has the specified annotation type."""
    return any(annotation_type == filter_type for annotation_type in doc["annotations"]["type"])

def process_docs(dataset: datasets.Dataset):
    """Process all documents."""
    return dataset.map(_process_doc_base)

def process_docs_ambiguous(dataset: datasets.Dataset):
    """Process only ambiguous documents (those with multiple QAs)."""
    # First filter for documents with multipleQAs, then process them
    filtered_dataset = dataset.filter(lambda doc: _has_annotation_type(doc, "multipleQAs"))
    return filtered_dataset.map(_process_doc_base)

def process_docs_nonambiguous(dataset: datasets.Dataset):
    """Process only non-ambiguous documents (those with single answers)."""
    # First filter for documents with singleAnswer, then process them
    filtered_dataset = dataset.filter(lambda doc: _has_annotation_type(doc, "singleAnswer"))
    return filtered_dataset.map(_process_doc_base)