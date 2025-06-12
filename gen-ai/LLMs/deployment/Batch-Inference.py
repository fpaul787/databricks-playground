# Databricks notebook source
# MAGIC %pip install accelerate

# COMMAND ----------

catalog_name = "frantzpaul_tech"
schema_name = "batch_xsum"

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare Dataset

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline
from delta.tables import DeltaTable


prod_data_table_name = f"{catalog_name}.{schema_name}.prod_data"

xsum_dataset = load_dataset(
    "xsum",
    version="1.2.0",
    trust_remote_code=True
)

# Save test set to delta table
test_spark_df = spark.createDataFrame(xsum_dataset["test"].to_pandas())
test_spark_df.write.mode("overwrite").saveAsTable(prod_data_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM frantzpaul_tech.batch_xsum.prod_data;

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a Hugging Face Pipeline
# MAGIC We will use T5 Text-To-Text Transfer Transformer

# COMMAND ----------

from transformers import pipeline

# Define the pipeline inference parameters - to be logged in mlflow as part of the model _metadata
hf_model_name = "t5-small"
min_length = 20
max_length = 40
truncation = True
do_sample = True
device_map = "auto"

cache_dir = "/dbfs/FileStore/hf_cache"

summarizer = pipeline(
  task="summarization",
  model=hf_model_name,
  min_length=min_length,
  max_length=max_length,
  truncation=truncation,
  do_sample=do_sample,
  device_map="cpu",
  model_kwargs={"cache_dir": cache_dir},
)

# COMMAND ----------

# MAGIC %md
# MAGIC Now examine the `summarizer` pipeline summarizing some text.

# COMMAND ----------

text_to_summarize = """ Barrington DeVaughn Hendricks (born October 22, 1989), known professionally as JPEGMafia (stylized in all caps as JPEGMAFIA), is an American rapper, singer, and record producer. Born in Flatbush, Brooklyn, he signed with Deathbomb Arc to release his debut studio album, Black Ben Carson (2016) and his second album Veteran (2018), which received widespread critical acclaim. """

summarized_text = summarizer(text_to_summarize)[0]['summary_text']
print(f"Summary:\n {summarized_text}")
print("========================================")
print(f"Original Document: {text_to_summarize}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Development and Registering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Track LLM Development with MLflow

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output

output = generate_signature_output(summarizer, text_to_summarize)
signature = infer_signature(text_to_summarize, output)
print(f"Signature:\n{signature}")

# Set experiment path
experiment_name = f"/Users/frantz@frantzpaul.tech/GenAi-As-Batch-Demo"
mlflow.set_experiment(experiment_name)
model_artifact_path = "summarizer" # Name of folder of serialized model

with mlflow.start_run() as run:
    # LOG PARAMS
    mlflow.log_params(
        {
            "hf_model_name": hf_model_name,
            "min_length": min_length,
            "max_length": max_length,
            "truncation": truncation,
            "do_sample": do_sample
        }
    )

    # -------
    # LOG MODEL
    inference_config = {
        "min_length": min_length,
        "max_length": max_length,
        "truncation": truncation,
        "do_sample": do_sample
    }

    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path=model_artifact_path,
        task="summarization",
        inference_config=inference_config,
        signature=signature,
        input_example="This is an example of a long news article which this pipeline can summarize for you"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the MLflow Tracking server
# MAGIC
# MAGIC MLflow Tracking API:
# MAGIC
# MAGIC MLflow Tracking UI:

# COMMAND ----------


