import os
import numpy as np
import json

import pandas as pd
import torch
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel


# Define a function to create a mapping from unique categories to their indices
def make_map(categories):
    return {x: i for i, x in enumerate(categories)}


# Define a function to pad and truncate a list of torch tensors
def pad_and_truncate(sequences, max_len, padding_value=-1):
    lengths = [min(len(x), max_len) for x in sequences]
    sequences = [torch.Tensor(x) for x in sequences]
    for i, tensor in enumerate(sequences):
        if tensor.dim() == 0:
            sequences[i] = torch.tensor([padding_value])
    sequences.append(torch.zeros(max_len))
    sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=padding_value
    )
    sequences = sequences[:-1, :max_len]
    sequences = sequences + 1
    lengths = torch.tensor(lengths).int().view(-1, 1)
    return sequences, lengths


def preprocess(recommendations, games):
    # Make copies of input dataframes to avoid modifying them directly
    recommendations = recommendations.copy()
    games = games.copy()

    # Cast app_id and user_id columns to integer type
    games["app_id"] = games["app_id"].astype("int")
    recommendations["app_id"] = recommendations["app_id"].astype("int")
    recommendations["user_id"] = recommendations["user_id"].astype("int")

    # Filter the recommendations dataframe to keep only the rows corresponding to games that are in the games dataframe
    recommendations = recommendations[recommendations["app_id"].isin(games["app_id"])]

    # Merge the resulting dataframe with the games dataframe to get the release dates of the games
    recommendations = pd.merge(
        recommendations, games[["app_id", "date_release"]], on="app_id"
    )

    # Filter the resulting dataframe to keep only the rows where the game is recommended and where the helpfulness of the review is greater than 1
    recommendations = recommendations.query("is_recommended == True & helpful > 1")

    # Filter the users to keep only those who have reviewed more than 1 game
    s = recommendations.groupby("user_id").count()["app_id"]
    filt = s[s > 1]
    recommendations = recommendations[recommendations["user_id"].isin(filt.index)]

    # Filter the games to only keep those which have at least 5 recommendations
    s = recommendations.groupby("app_id").count()["user_id"]
    filt = s[s > 5]
    recommendations = recommendations[recommendations["app_id"].isin(filt.index)]

    # Filter games to keep only those in recommendations
    games = games[games["app_id"].isin(recommendations["app_id"])]

    # Return filtered recommendations and games dataframes
    return recommendations, games


def create_app_user_pairs(recommendations):
    # Group the recommendations dataframe by mapped user id and mapped app id, and count the number of occurrences
    user_app_pairs = (
        recommendations.groupby(["mapped_userid", "mapped_appid"])
        .size()
        .unstack(fill_value=0)
    )

    # Convert the resulting dataframe to a sparse matrix
    user_app_pairs = user_app_pairs.astype(pd.SparseDtype("int", 0))
    user_app_pairs = user_app_pairs.sparse.to_coo()

    return user_app_pairs


def get_tensor_from_category_enumerations(features, col, seq_len, MAPPINGS):
    """
    Converts a column of categorical data into a tensor of integers representing the categories.
    """
    # Get the set of all categories in the column
    categories = set()
    categories.update(*list(features[col].str.split(",")))

    # Create a map of each category to a unique integer
    map_categories = make_map(categories)
    MAPPINGS |= {col: map_categories}

    # Convert each entry in the column into a list of integers representing the categories, then pad and truncate the
    # lists so they are all the same length
    categories_vec = list(
        features[col].str.split(",").apply(lambda l: [map_categories[x] for x in l])
    )
    categories_tensor = pad_and_truncate(categories_vec, seq_len)
    return categories_tensor, MAPPINGS


# Preprocesses the "About the game" section
def preprocess_about_game(features):
    print("Loading FastText model...")
    fasttext_model = gensim.models.fasttext.load_facebook_model("wiki.en/wiki.en.bin")
    print("Loading done!")

    documents = list(features["About the game"])

    # Remove Nones for documents
    documents = ["None" if doc is None else doc for doc in documents]

    # Tokenize
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Create a dictionary from the tokenized documents
    dictionary = Dictionary(tokenized_docs)

    # Create a bag of words (BoW) representation for each document
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Fit the TfidfModel using the BoW corpus
    tfidf_model = TfidfModel(corpus)

    def get_tfidf_weighted_embeddings(tfidf_model, corpus, fasttext_model, dictionary):
        weighted_embeddings = []

        for doc_bow in corpus:
            tfidf_scores = dict(tfidf_model[doc_bow])
            doc_embedding = np.zeros(fasttext_model.vector_size)
            total_weight = 0

            for word_id, tfidf_score in tfidf_scores.items():
                word = dictionary[word_id]
                if word in fasttext_model.wv:
                    word_embedding = fasttext_model.wv[word]
                    doc_embedding += word_embedding * tfidf_score
                    total_weight += tfidf_score

            if total_weight > 0:
                doc_embedding /= total_weight

            weighted_embeddings.append(doc_embedding)

        return np.array(weighted_embeddings)

    weighted_embeddings = get_tfidf_weighted_embeddings(
        tfidf_model, corpus, fasttext_model, dictionary
    )

    weighted_embeddings = torch.Tensor(weighted_embeddings)
    return weighted_embeddings


def main():
    # Define the path to the "data" directory
    data_dir = "data"

    # Load the Parquet files
    descriptions = pd.read_parquet(os.path.join(data_dir, "descriptions.parquet"))
    recommendations = pd.read_parquet(os.path.join(data_dir, "recommendations.parquet"))
    games = pd.read_parquet(os.path.join(data_dir, "games.parquet"))

    # Rename the "AppID" column in the descriptions dataframe to "app_id"
    descriptions = descriptions.rename(columns={"AppID": "app_id"})

    # Merge the games dataframe with the descriptions dataframe on the "app_id" column
    games = pd.merge(games, descriptions, on="app_id", how="inner")

    MAPPINGS = {}

    # Preprocess the dataframes
    recommendations, games = preprocess(recommendations, games)

    # Map app and user ids to ranges
    MAPPINGS |= {
        "app": make_map(
            list(recommendations.sort_values("date_release")["app_id"].unique())
        )
    }
    MAPPINGS |= {"user": (make_map(list(recommendations["user_id"].unique())))}

    # Apply mappings to create new columns in recommendations and games dataframes
    recommendations["mapped_appid"] = recommendations["app_id"].map(MAPPINGS["app"])
    recommendations["mapped_userid"] = recommendations["user_id"].map(MAPPINGS["user"])
    games["mapped_appid"] = games["app_id"].map(MAPPINGS["app"])

    # Call the create_app_user_pairs function to get a sparse matrix of user-app pairs
    user_app_pairs = create_app_user_pairs(recommendations)

    # Multiply the user-app pairs matrix by its transpose to get an adjacency matrix of app-app pairs
    edges = user_app_pairs.T.dot(user_app_pairs)

    # Convert the edges matrix to a dense numpy array
    edges = edges.toarray()

    # Threshold the edges matrix so that only edges with weight greater than the threshold are kept
    # Set the diagonal values to 0 to remove self-references
    np.fill_diagonal(edges, 0)
    edges = edges.dot(np.diag(1 / edges.sum(axis=1)))
    np.fill_diagonal(edges, 1)

    # Define a list of column names to use as features for each app
    features_cols = [
        "mapped_appid",
        "title",
        "date_release",
        "price_original",
        "Developers",
        "Genres",
        "Tags",
        "About the game",
    ]

    # Create a dataframe of app features using the games dataframe and the features_cols list
    features = games[features_cols].set_index("mapped_appid").sort_index()

    ## Price
    # Divide the price by 5 and round to the nearest integer to create a new _price feature
    price_tensor = (features["price_original"] / 5).round().astype(int).values
    price_tensor = torch.Tensor(price_tensor).view(-1, 1)

    ## Developer
    # Map the Developers column to integers using the MAPPINGS dictionary, and store the result in a new _developer feature
    MAPPINGS |= {"developer": make_map(features["Developers"].unique())}
    developer_tensor = features["Developers"].map(MAPPINGS["developer"]).values
    developer_tensor = torch.Tensor(developer_tensor).view(-1, 1)

    ## Genres
    # Clean up the Genres column by removing the "Early Access" tag from later games
    features["Genres"] = features["Genres"].str.replace(",Early Access", "")

    # Get Tensor from genres enumerations
    from params import GENRES_SEQ_LEN

    genres_tensor, MAPPINGS = get_tensor_from_category_enumerations(
        features, "Genres", GENRES_SEQ_LEN, MAPPINGS
    )

    ## Tags
    # Get Tensor from tags enumerations
    from params import TAGS_SEQ_LEN

    # Replace null values in the tags column with a placeholder string
    features["Tags"] = features["Tags"].fillna("NoTags")
    tags_tensor, MAPPINGS = get_tensor_from_category_enumerations(
        features, "Tags", TAGS_SEQ_LEN, MAPPINGS
    )

    ## About the Game
    # Clearing up some memory because the FastText embedding is huge
    del descriptions
    del games
    del recommendations

    weighted_embeddings = preprocess_about_game(features)

    # Combine the feature tensors into a single tensor
    X = torch.cat(
        (
            price_tensor,
            developer_tensor,
            *genres_tensor,
            *tags_tensor,
            weighted_embeddings,
        ),
        dim=1,
    )
    # size(1) =  1 + 1 + (GENRES_SEQ_LEN + 1) + (TAGS_SEQ_LEN + 1) + 300(fasttext size)

    ## Write to disk
    # Define the output directory for the processed data
    output_dir = "run_artifacts/preprocess"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the edges matrix to disk
    pd.DataFrame(edges).to_csv(f"{output_dir}/edges.csv")

    # Convert the app and user mappings to dictionaries with integer keys
    MAPPINGS["app"] = {int(a): int(b) for a, b in enumerate(MAPPINGS["app"])}
    MAPPINGS["user"] = {int(a): int(b) for a, b in enumerate(MAPPINGS["user"])}

    # Save the mappings to disk

    with open(f"{output_dir}/mappings.json", "w") as f:
        json.dump(MAPPINGS, f)

    # Save the feature tensor and the edges tensor as a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(X, torch.Tensor(edges))
    torch.save(dataset, f"{output_dir}/dataset.t")


if __name__ == "__main__":
    main()
