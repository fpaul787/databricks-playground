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

# MAGIC %md
# MAGIC ### T5-base
# MAGIC The T5-baes model 220 million parameters.

# COMMAND ----------

t5_base_summaries = summarize_with_t5(
    model_checkpoint="t5-base", articles=sample["article"]
)
compute_rouge_score(t5_base_summaries, reference_summaries)

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPT-2
# MAGIC The GPT-2 model is a generative text model that was trained in a self-supervised fashion. 

# COMMAND ----------

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def summarize_with_gpt2(model_checkpoint: str, articles: list, batch_size: int = 8) -> list:
  """
  Funtion to summarize articles with GPT2 to handle these complications:
  - Append "TL:DR" ot the end of the input to get GPT2 to generate a summary
  - GPT2 uses a max token length of 1024. We use a shorter 512 limit
  """
  if torch.cuda.is_available():
    device = "cuda:0"
  else:
    device = "cpu"

  tokenizer = GPT2Tokenizer.from_pretrained(
    model_checkpoint, padding_side="left", cache_dir="/dbfs/tmp/llm_cache"
  )

  tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
  model = GPT2LMHeadModel.from_pretrained(
    model_checkpoint,
    pad_token_id=tokenizer.eos_token_id,
    cache_dir="/dbfs/tmp/llm_cache"
  ).to(device)

  def perform_inference(batch: list) -> list:
    tmp_inputs = tokenizer(
      batch, max_length=500, return_tensors="pt", padding=True, truncation=True
    )

    tmp_inputs_decoded = tokenizer.batch_decode(
      tmp_inputs.input_ids, skip_special_tokens=True
    )

    inputs = tokenizer(
      [article + " TL:DR" for article in tmp_inputs_decoded],
      max_length=512,
      return_tensors="pt",
      padding=True
    )

    summary_ids = model.generate(
      inputs.input_ids.to(device),
      attention_mask=inputs.attention_mask.to(device),
      max_length=512 + 32,
      num_beams=2,
      min_length=0
    )

    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

  decoded_summaries = []
  for batch in batch_generator(articles, batch_size=batch_size):

    decoded_summaries += perform_inference(batch)

    # batch clean up
    torch.cuda.empty_cache()
    gc.collect()

  # post-process decoded summaries
  summaries = [
    summary[summary.find("TL;DR") + len("TL;DR: ") :]
    for summary in decoded_summaries
  ]

  # cleanup
  del tokenizer
  del model
  torch.cuda.empty_cache()
  gc.collect()

  return summaries

# COMMAND ----------

gpt2_summaries = summarize_with_gpt2(
    model_checkpoint="gpt2", articles=sample["article"]
)

compute_rouge_score(gpt2_summaries, reference_summaries)

# COMMAND ----------

gpt2_results = compute_rouge_per_row(
    generated_summaries=gpt2_summaries, reference_summaries=reference_summaries
)
display(gpt2_results)

# COMMAND ----------

def compare_models(model_results: dict) -> pd.DataFrame:
    agg_results = []

    for r in models_results:
        model_results = model_results[r].drop(
            labels=["generated", "reference"], axis=1
        )
        agg_metrics = [r]
        agg_metrics[1:] = model_results.mean(axis=0)
        agg_results.append(agg_metrics)

    return pd.DataFrame(
        agg_results, columns=["model", "rouge1", "rouge2", "rougeL", "rougeLsum"]
    )

# COMMAND ----------

display(
    compare_models(
        {
            "t5-small": t5_small_results,
            "t5-base": t5_base_results,
            "gpt2": gpt2_results
        }
    )
)

# COMMAND ----------

def compare_models_summaries(models_summaries: dict) -> pd.DataFrame:
    """
    Aggregates results from `models_summaries` and returns a dataframe
    """
    comparison_df = None
    for model_name in models_summaries:
        summaries_df = models_summaries[model_name]
        if comparison_df is None:
            comparison_df = summaries_df[["generated"]].rename(
                {"generated": model_name}, axis=1
            )
        else:
            comparison_df[model_name] = comparison_df.join(
                summaries_df[["generated"]].rename(
                    {"generated": model_name}, axis=1
            )
    return comparison_df

# COMMAND ----------

display(
    compare_model_summaries(
        {
            "t5-small": t5_small_summaries,
            "t5-base": t5_base_summaries,
            "gpt2": gpt2_summaries,
        }
    )
)
