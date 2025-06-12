# Databricks notebook source
# MAGIC %md
# MAGIC # Model Preparation

# COMMAND ----------

import mlflow
from mlflow import MlFlowClient

catalog_name = "frantzpaul_tech"
schema_name = "dbdemos_rag_chatbot"

model_name = f"{catalog_name}.{schema_name}.rag_app" # Does not exist yet

# Point to UC registry
mlflow.set_registry_uri("databricks-uc")

def get_latest_model_version(model_name_in:str = None):
  """
  Get latest version of registered model
  """
  client = MlflowClient()
  model_version_infos = client.search_model_version("name = '%s'" % model_name_in)
  if model_version_infos:
    return max([model_version_info.version for model_version_info in model_version_infos])
  else:
    return None
