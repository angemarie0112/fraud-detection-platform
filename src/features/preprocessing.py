from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def create_preprocessor():
    """
    Create a preprocessing transformer for fraud detection.

    Standardizes Time and Amount while preserving
    the anonymized PCA features V1–V28.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scale",
                StandardScaler(),
                ["Time", "Amount"]
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    return preprocessor