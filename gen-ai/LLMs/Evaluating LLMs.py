# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluation Metrics

# COMMAND ----------

import torch
from datasets import load_dataset

full_dataset = load_dataset(
  "cnn_dailymail", "3.0.0", cache_dir="/dbfs/databricks/datasets"
)
