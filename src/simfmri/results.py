"""Gather and analyse hydra results."""

import argparse
import pandas as pd
import seaborn as sns

from simfmri.glm import compute_stats_df


def roc_curve():
    """Plot a Receive Operator Characteristic."""
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+")

    ns = parser.parse_args()

    df_list = []
    for file in ns.result_files:
        try:
            df_list.append(pd.read_csv(file))
        except FileNotFoundError:
            print(file + " not found")

    result_df = pd.concat(df_list)

    stat_df = compute_stats_df(result_df)

    roc_curve(stat_df)


if __name__ == "__main__":
    main()
