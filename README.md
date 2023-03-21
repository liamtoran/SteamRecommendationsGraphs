# Description

This repository contains a machine learning / data science project whose goal is to predict game recommendations using data scraped from the Steam API, the largest distribution platform for PC gaming.

- The `preprocessing.py` code reads, filters and cleans the data, and creates a PyTorch dataset that can be used in a machine learning model.
- The `train_model.py` code contains the machine learning model itself. The Model class is a PyTorch neural network that takes in various features about a game (such as price, developer, genres, tags, and a weighted fasttext embedding of the "About the game" text description in steam) and outputs a vector of recommendations.


# How to Use

- Clone the repository.
- Get the data (ask me)
- In the terminal, navigate to the repository directory.
- To preprocess the data, do `python src/preprocess.py` The preprocessed data will be saved in a new directory called `run_artifacts/preprocess`
- To train the model, do `python train_model.py` file. The model will be trained on the preprocessed data in the run_artifacts/preprocess directory and saved in a new directory called `run_artifacts/train_model`.