import matplotlib.pyplot as plt
import pandas as pd


def plot_propensity_distribution(
    df: pd.DataFrame,
    propensity_col: str = "propensity_score",
    treatment_col: str = "T",
    bins: int = 30,
):
    """
    График overlap propensity score.
    """

    treated = df.loc[df[treatment_col] == 1, propensity_col]
    control = df.loc[df[treatment_col] == 0, propensity_col]

    plt.figure(figsize=(10, 6))

    plt.hist(
        control,
        bins=bins,
        alpha=0.6,
        label="Control",
        density=True,
    )

    plt.hist(
        treated,
        bins=bins,
        alpha=0.6,
        label="Treatment",
        density=True,
    )

    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.title("Propensity Score Distribution")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_uplift_distribution(
    df: pd.DataFrame,
    uplift_col: str = "uplift",
    bins: int = 40,
):
    """
    Распределение uplift prediction.
    """

    plt.figure(figsize=(10, 6))

    plt.hist(
        df[uplift_col],
        bins=bins,
    )

    plt.xlabel("Predicted Uplift")
    plt.ylabel("Count")
    plt.title("Uplift Distribution")

    plt.tight_layout()
    plt.show()


def plot_bootstrap_distribution(
    bootstrap_values,
    ci_lower: float,
    ci_upper: float,
):
    """
    Распределение bootstrap incremental revenue.
    """

    plt.figure(figsize=(10, 6))

    plt.hist(
        bootstrap_values,
        bins=40,
    )

    plt.axvline(ci_lower, linestyle="--")
    plt.axvline(ci_upper, linestyle="--")

    plt.xlabel("Incremental Revenue")
    plt.ylabel("Count")
    plt.title("Bootstrap Distribution")

    plt.tight_layout()
    plt.show()