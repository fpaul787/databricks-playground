# Databricks notebook source
# MAGIC %md
# MAGIC # Building Multi-stage Reasoning Systems with LangChain
# MAGIC Two AI systems:
# MAGIC * The first, codenamed `Fhyde` will be a prototype AI self-commenting -and-moderating tool that will create new reaction comments to a piece of text withi one LLM and use another LLM tocritique those comments and flag them if they are negative. 
# MAGIC * The second system, condenamed `Stacey` will take the form of an LLM-based agent that will be tasked with performing data science tasks on data that will be stored in a vector database using ChromaDB. This will be made using LangChain agents as well as the ChromaDB library, as well as the Pandas Dataframe Agent and python REPL tool.

# COMMAND ----------

# MAGIC %pip install wikipedia==1.4.0 google-search-results==2.4.2 better-profanity==0.7.0
