"""
In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations. The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution.

Addressing this need, we present Unitxt, an innovative library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. The Unitxt-Catalog centralizes these components, fostering collaboration and exploration in modern textual data workflows. Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively.
"""

import importlib.util
import re
from collections.abc import Callable
from functools import partial
from typing import Any, Dict, Optional

import datasets

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


_CITATION = """
@misc{bandel2024unitxt,
      title={Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI},
      author={Elron Bandel and Yotam Perlitz and Elad Venezian and Roni Friedman-Melamed and Ofir Arviv and Matan Orbach and Shachar Don-Yehyia and Dafna Sheinwald and Ariel Gera and Leshem Choshen and Michal Shmueli-Scheuer and Yoav Katz},
      year={2024},
      eprint={2401.14019},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


def assert_unitxt_installed():
    if importlib.util.find_spec("unitxt") is None:
        raise ModuleNotFoundError(
            "Please install unitxt via 'pip install unitxt'. For more information see: https://www.unitxt.ai/"
        )


def score(items, metric):
    predictions, references = zip(*items)
    assert_unitxt_installed()
    from unitxt import evaluate

    for reference in references:
        reference["metrics"] = [metric]
    results = evaluate(predictions, references)
    return results[0]["score"]["global"]["score"]


class Unitxt(ConfigurableTask):
    VERSION = 0

    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        if config is None:
            config = {}
        assert "recipe" in config, "Unitxt task must have a 'recipe' string."
        
        # Extract repeats and generation_kwargs from the original config
        self.repeats_value = config.get("repeats", 1)
        self.generation_kwargs = config.get("generation_kwargs", {"until": ["\n"]})
        
        # Create the configuration dict with all necessary parameters
        task_config = {
            "metadata": {"version": self.VERSION},
            "dataset_name": config["recipe"],
            "metric_list": [
                {
                    "metric": "unitxt",
                    "aggregation": "mean",
                    "higher_is_better": True
                }
            ],
            "repeats": self.repeats_value,
            "generation_kwargs": self.generation_kwargs
        }
        
        super().__init__(config=task_config)
        self.image_decoder = datasets.Image()
        self.metrics = self.dataset["test"][0]["metrics"]
        
        # Track if we're doing the xsum task, used for special response handling
        self.is_xsum_task = "xsum" in config.get("task", "")

    def download(self, dataset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        assert_unitxt_installed()
        from unitxt import load_dataset

        # Parse the recipe string into components
        recipe_parts = {}
        for part in self.DATASET_NAME.split(','):
            key, value = part.split('=')
            recipe_parts[key.strip()] = value.strip()
        
        # Load dataset using the parsed recipe
        self.dataset = load_dataset(**recipe_parts)
        self.metrics = self.dataset["test"][0]["metrics"]

    def has_training_docs(self):
        return "train" in self.dataset

    def has_validation_docs(self):
        return "validation" in self.dataset

    def has_test_docs(self):
        return "test" in self.dataset

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["source"]

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        doc["target"]

    def get_arguments(self, doc, ctx):
        # Use the generation_kwargs from config for arguments
        return (ctx, self.generation_kwargs)

    def fewshot_context(
        self,
        doc: str,
        num_fewshot: int,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Optional[Callable] = None,
        gen_prefix: Optional[str] = None,
    ) -> str:
        source = self.doc_to_text(doc)
        if isinstance(source, list):
            if apply_chat_template:
                formated_source = chat_template(self.doc_to_text(doc))
                return formated_source
            else:
                raise Exception(
                    "Got chat template format from Unitxt, but apply_chat_template is false. Add '--apply_chat_template' to command line."
                )
        else:
            return source

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        kwargs.pop("apply_chat_template", False)  # Not used by unitxt
        
        # Get metadata and remove it from kwargs to avoid passing it twice
        task_name, doc_id, _ = kwargs.pop("metadata", (None, None, None))
        
        # For xsum, create just one instance with repeats=10 (or whatever the repeats value is)
        # This way, we get all responses in a single request
        if "xsum" in task_name:
            new_metadata = (task_name, doc_id, self.repeats_value)
            return [
                Instance(
                    request_type="generate_until",
                    doc=doc,
                    arguments=self.get_arguments(doc, ctx),
                    idx=0,
                    metadata=new_metadata,
                    **kwargs,
                )
            ]
        
        # For other tasks, create multiple separate instances
        instances = []
        for i in range(self.repeats_value):
            # Create a separate instance for each repeat with repeats=1
            new_metadata = (task_name, doc_id, 1)
            instances.append(
                Instance(
                    request_type="generate_until",
                    doc=doc,
                    arguments=self.get_arguments(doc, ctx),
                    idx=i,  # Use different idx for each repeat
                    metadata=new_metadata,
                    **kwargs,
                )
            )
        
        return instances

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # For xsum tasks with repeats, the results might be nested
        # Extract the first response for evaluation metrics
        if self.is_xsum_task and isinstance(results, list) and len(results) == 1:
            prediction = results[0][0] if results[0] else ""
        else:
            prediction = results[0] if results else ""

        references = doc
        return {
            metric.replace("metrics.", ""): (prediction, references)
            for metric in self.metrics
        }
    
    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            metric.replace("metrics.", ""): partial(score, metric=metric)
            for metric in self.metrics
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {metric.replace("metrics.", ""): True for metric in self.metrics}


images_regex = r'<img\s+src=["\'](.*?)["\']\s*/?>'
image_source_regex = r'<img\s+src=["\'](.*?)["\']'


def extract_images(text, instance):
    image_sources = re.findall(image_source_regex, text)
    images = []
    for image_source in image_sources:
        current = instance
        for key in image_source.split("/"):
            if key.isdigit():
                key = int(key)
            current = current[key]
        images.append(current)
    return images


class UnitxtMultiModal(Unitxt):
    MULTIMODAL = True

    def doc_to_text(self, doc):
        return re.sub(images_regex, "<image>", doc["source"])

    def doc_to_image(self, doc):
        images = extract_images(doc["source"], doc)
        return [self.image_decoder.decode_example(image) for image in images]

    def get_arguments(self, doc, ctx):
        args = super().get_arguments(doc, ctx)
        visual_args = {"visual": self.doc_to_image(doc)}
        
        # Check if args is a tuple with two elements (context, kwargs)
        if isinstance(args, tuple) and len(args) == 2:
            return (args[0], args[1], visual_args)
        return (ctx, self.generation_kwargs, visual_args)
