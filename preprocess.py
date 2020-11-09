import pandas as pd
import random 
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument(
        "--source_articles",
        type=str,
        default=os.path.join("./", "data/raw_data/train/train.article.txt"),
        help="Path to the articles' source file to preprocess"
    )
    parser.add_argument(
        "--source_titles",
        type=str,
        default=os.path.join("./", "data/raw_data/train/train.title.txt"),
        help="Path to the titles' source file to preprocess"
    )
    parser.add_argument(
        "--destination",
        type=str,
        default=os.path.join("./", "data/processed_data/train.csv"),
        help="Path to the output csv file"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size"
    )
    opts = parser.parse_args()

    data = pd.concat([
        pd.read_csv(opts.source_articles, sep="\n"), 
        pd.read_csv(opts.source_titles, sep="\n")], axis=1)
    data.columns = ["article", "title"]

    data = data.sample(opts.sample)

    data.to_csv(opts.destination, index=None)
