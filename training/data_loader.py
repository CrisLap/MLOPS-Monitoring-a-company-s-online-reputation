from datasets import load_dataset


def load_data():
    """
    Loads the TweetEval sentiment dataset.
    Returns the dataset object containing
    training, validation, and test splits.
    """
    dataset = load_dataset("tweet_eval", "sentiment")
    return dataset
