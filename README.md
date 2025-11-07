# Project: Advanced Information Retrieval System

This project is an advanced information retrieval (IR) system developed as part of our university curriculum. It provides a graphical user interface to explore and evaluate various IR models and text processing techniques. The system is built using Python and leverages the `tkinter` library for the interface and `nltk` for natural language processing.

## Authors

*   **TAOURIRT Hamza**
*   **Sebti Asma**

## Table of Contents

1.  Project Description
2.  Core Features
3.  Technical Architecture
    *   Preprocessing and Indexing
    *   Search and Retrieval Models
    *   Evaluation Framework
4.  Technologies Used
5.  Setup and Usage
    *   Prerequisites
    *   Running the Application
    *   Using the Interface
6.  File Structure

## Project Description

This application allows users to perform searches on a document collection using several classic information retrieval models. It is designed to be both a practical search tool and an educational platform for comparing the effects of different preprocessing pipelines and retrieval algorithms.

Users can input free-text or boolean queries, select tokenization and stemming methods, and choose from Vector Space, Probabilistic (BM25), or Boolean retrieval models. The system also features a robust evaluation mode that calculates standard IR metrics like Precision, Recall, and F-score, and visualizes performance with a Recall/Precision curve.

## Core Features

*   **Graphical User Interface**: An intuitive UI built with `tkinter` for easy interaction.
*   **Flexible Preprocessing**:
    *   **Tokenization**: Choice between simple whitespace splitting (`Split`) or a more robust `RegexpTokenizer`.
    *   **Normalization**: Optional stemming with `PorterStemmer` or `LancasterStemmer`.
*   **Multiple Retrieval Models**:
    *   **Vector Space Model**: Supports three similarity measures:
        *   Scalar Product
        *   Cosine Similarity
        *   Jaccard Coefficient
    *   **Probabilistic Model**: Implements the **BM25** ranking function with configurable `k` and `b` parameters.
    *   **Boolean Model**: Processes queries with `AND`, `OR`, and `NOT` operators, respecting operator precedence.
*   **Comprehensive Evaluation Module**:
    *   Uses a predefined set of queries and relevance judgements (`Queries` and `Judgements` files).
    *   Calculates and displays **Precision**, **Recall**, **F-score**, **P@5**, and **P@10**.
    *   Generates and displays an interpolated **Recall/Precision curve** for visual performance analysis.
*   **Pre-computed Indexes**: The system uses pre-generated inverted files and descriptor files to ensure fast query processing.

## Technical Architecture

The project is divided into two main parts: an offline indexing script and the main online search interface.

### Preprocessing and Indexing

The `Create_Documents_Desc_Inv.py` script is responsible for offline processing. It reads the raw text documents from the `Collections/` directory and performs the following steps:
1.  **Tokenization and Stop Word Removal**: It processes the text to generate a list of tokens, removing common English stop words.
2.  **Frequency and Weight Calculation**: For each term in each document, it calculates the term frequency (TF) and the normalized TF-IDF weight. The weight is calculated as: `poids = (freq / max_freq) * log10(N / n_i + 1)`.
3.  **Stemming**: It applies Porter and Lancaster stemmers to the tokens and recalculates frequencies and weights for the stemmed terms.
4.  **Index Generation**: It generates two types of files for each configuration (tokenization/stemming combination):
    *   **Descriptor Files (`Descripteur*.txt`)**: Stores `(doc_id, token, frequency, weight)` tuples, sorted by document ID. Used for models requiring document-term information.
    *   **Inverted Files (`Inverse*.txt`)**: Stores `(token, doc_id, frequency, weight)` tuples, sorted by token. This is the classic inverted index structure.

### Search and Retrieval Models

The `Interface.py` file contains the logic for the GUI and the retrieval models. When a user executes a search:
1.  The query is preprocessed using the same tokenization and stemming settings selected in the UI.
2.  The appropriate pre-computed index file is loaded based on the UI settings.
3.  The selected retrieval model is executed:
    *   **Vector/Probabilistic Models**: Iterate through documents, calculate a relevance score (RSV) for each, and return a ranked list.
    *   **Boolean Model**: Parses the query logic and returns an unranked set of documents that strictly satisfy the boolean conditions.
4.  The results are displayed in the results table.

### Evaluation Framework

When the "Queries Dataset" option is enabled, the system enters evaluation mode:
1.  A query is selected from the `Queries` file using the spinbox.
2.  The corresponding set of known relevant documents is loaded from the `Judgements` file.
3.  The search is performed, and the retrieved document list is compared against the relevance judgements.
4.  Metrics are calculated and displayed, and the Recall/Precision graph is plotted using `matplotlib`.

## Technologies Used

*   **Language**: Python 3
*   **GUI**: `tkinter`
*   **NLP**: `nltk` (Natural Language Toolkit)
*   **Scientific Computing**: `numpy`
*   **Plotting**: `matplotlib`

## Setup and Usage

### Prerequisites

Ensure you have Python 3 installed. Then, install the required libraries:

```sh
pip install nltk numpy matplotlib
