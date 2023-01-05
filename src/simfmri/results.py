"""Gather and analyse hydra results."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from simfmri.glm import append_stats_df


def roc_curve(df: pd.DataFrame) -> None:
    """Plot a Receive Operator Characteristic."""
    pass

    # Draw a scatter plot while assigning point colors and sizes to different
    # variables in the dataset
    f, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(f, left=True, bottom=True)
    clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    return sns.scatterplot(
        x="TPR",
        y="TNR",
        hue="clarity",
        size="depth",
        palette="ch:r=-.2,d=.3_r",
        hue_order=clarity_ranking,
        data=df,
        ax=ax,
    )


def result_analysis(filenames: list[str]) -> None:
    """Main function for results analysis."""
    df_list = []
    for file in filenames:
        try:
            df_list.append(pd.read_csv(file, index_col=0))
        except FileNotFoundError:
            print(file + " not found")

    result_df = pd.concat(df_list)
    stat_df = append_stats_df(result_df)
    return stat_df


def main() -> None:
    """Main function for results analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+")

    ns = parser.parse_args()
    return result_analysis(ns.results_files)


if __name__ == "__main__":
    main()
