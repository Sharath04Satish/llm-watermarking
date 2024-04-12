from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


def getNewsArticles(dataset_name, sub_type, model_name, num_examples):
    realnewslike = load_dataset(dataset_name, sub_type, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = realnewslike.map(lambda examples: tokenizer(examples["text"], return_tensors="np"), batched=True)["train"]

    i = np.random.uniform(low=0, high=100, size=(1)).astype(int)[0]
    j = i + num_examples

    news = list()

    for example in dataset:
        i += 1
        if i < j:
            news.append(
                {
                    "text": example["text"],
                }
            )
        else:
            break

    return news
