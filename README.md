# Obscura: Building an LLM from Scratch

This repository documents the journey of building a Large Language Model (LLM) from scratch, inspired by the concepts presented in the "Attention Is All You Need" paper.

## Chapter 1: Initial Setup and Dependencies

This initial commit focuses on setting up the development environment and installing necessary dependencies.

### Dependencies Installed:

* `uv`: A fast Python package installer and resolver.

## Chapter 2: Working with Text Data

This chapter begins the process of preparing text data for model training.

### Initial Text Processing:

* **Data Loading:** Loaded the raw text content from the "Attention Is All You Need" paper.
* **File Storage:** Saved the raw text content to `sample_data.txt` for local access.
* **Basic Tokenization Exploration:** Explored initial methods for splitting text into tokens using regular expressions, handling spaces, commas, and periods.
* **Token Cleaning:** Refined the tokenization process to strip leading/trailing whitespace from tokens and filter out any empty strings resulting from the split.
* **Expanded Punctuation:** Extended the set of punctuation characters considered during splitting to include more common symbols.
* **Application to Raw Text:** Applied the developed tokenization logic to the full `sample_data.txt`.
