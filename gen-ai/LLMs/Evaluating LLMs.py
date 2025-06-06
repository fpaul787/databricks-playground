# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset
# MAGIC
# MAGIC `cnn_dailymail` dataset from See et al. 2017, from HuggingFace datasets hub. This dataset provides news article paired with summaries (in the "highlights" column).

# COMMAND ----------

import torch
from datasets import load_dataset

full_dataset = load_dataset(
  "cnn_dailymail", "3.0.0", cache_dir="/dbfs/databricks/datasets"
)

sample_size = 100
sample = (
  full_dataset["train"]
    .filter(lambda r: "CNN" in r["article"][:25])
    .shuffle(seed=42)
    .select(range(sample_size))
)
sample

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summarization

# COMMAND ----------

import pandas as pd
import torch
import gc
from transformers import AutoTokenizer, T5ForConditionalGeneration

def batch_generator(data: list, batch_size: int):
    """
    Creates batches of size `batch_size` from a list to feed to model
    """
    s = 0
    e = s + batch_size
    while s < len(data):
        yield data[s:e]
        s = e
        e = min(s + batch_size, len(data))

def summarize_with_t5(
    model_checkpoint: str, articles: list, batch_size: int = 8
) -> list:
    """
    Compute summaries using a T5 model.
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    model = T5ForConditionalGeneration.from_pretrained(
        model_checkpoint, cache_dir="/dbfs/databricks/datasets"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, model_max_length=1024, cache_dir="/dbfs/databricks/datasets"
    )

    def perform_inference(batch: list) -> list:
        inputs = tokenizer(
            batch, max_length=1024, return_tensors="pt", padding=True, truncation=True
        )

        summary_ids = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            num_beams=2,
            min_length=0,
            max_length=40
        )
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    res = []
    
    summary_articles = list(map(lambda article: "summarize: " + article, articles))
    for batch in batch_generator(summary_articles, batch_size=batch_size):
        res += perform_inference(batch)

        torch.cuda.empty_cache()
        gc.collect()

    # clean up
    del tokenizer
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return res

# COMMAND ----------


t5_small_summaries = summarize_with_t5("t5-small", sample["article"])
reference_summaries = sample["highlights"]

# COMMAND ----------

display(
    pd.DataFrame.from_dict(
        {
            "generated": t5_small_summaries,
            "reference": reference_summaries
        }
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROUGE
# MAGIC Recall-Oriented Understudy for Gisting Evaluation (ROUGE) is a set of evaluation metrics designed fro comparing summaries from Lin et al., 2004. 

# COMMAND ----------

# MAGIC %pip install rouge_score

# COMMAND ----------

import evaluate
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

rouge_score = evaluate.load("rouge")

# COMMAND ----------

def compute_rouge_score(generated: list, reference: list) -> dict:
    """
    Compute ROUGE scores on a batch of articles

    output dict with scores
    """

    generated_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in generated]
    reference_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in reference]
    return rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True
    )

# COMMAND ----------

compute_rouge_score(t5_small_summaries, reference_summaries)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understaning ROUGE scores

# COMMAND ----------

compute_rouge_score(reference_summaries, reference_summaries)

# COMMAND ----------

compute_rouge_score(
    generated=["" for _ in range(len(reference_summaries))],
    reference=reference_summaries
)

# COMMAND ----------

rouge_score.compute(
    predictions=["Large"],
    references=["Large language models beat world record"],
    use_stemmer=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare small and large models

# COMMAND ----------

def compute_rouge_per_row(
    generated_summaries: list, reference_summaries: list
) -> pd.DataFrame:
    """
    Generates a dataframe to compare rogue score metrics.
    """
    generated_with_newlines = [
        "\n".join(sent_tokenize(s.strip())) for s in generated_summaries
    ]
    reference_with_newlines = [
        "\n".join(sent_tokenize(s.strip())) for s in reference_summaries
    ]
    scores = rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True,
        use_aggregator=False
    )
    scores["generated"] = generated_summaries
    scores["reference"] = reference_summaries
    return pd.DataFrame.from_dict(scores)

# COMMAND ----------

compute_rouge_score(t5_small_summaries, reference_summaries)

# COMMAND ----------

t5_small_results = compute_rouge_per_row(
    generated_summaries=t5_small_summaries, reference_summaries=reference_summaries
)
display(t5_small_results)

# COMMAND ----------


