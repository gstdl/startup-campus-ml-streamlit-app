from typing import Dict, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .woe_transformer import WoeTransformer

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/raw/telco_churn.csv")

# selected_features = [
#     "InternetService_WoE",
#     "Contract_WoE",
#     "tenure",
#     "TotalCharges",
#     "MultipleLines_OH_No",
#     "OnlineSecurity_OH_Yes",
#     "PaymentMethod_OH_Electronic check",
# ]

pre_selected_features = [
    "InternetService",
    "Contract",
    "tenure",
    "TotalCharges",
    "MultipleLines",
    "OnlineSecurity",
    "PaymentMethod",
]

X, y = df[pre_selected_features], df["Churn"]
X["TotalCharges"] = X["TotalCharges"].str.replace(" ", "0").astype(np.float64)
wt = WoeTransformer(X, y, {"No": "Events", "Yes": "Non events"})
wt.single_fit("InternetService")
wt.single_fit("Contract")

standard_scaler = StandardScaler()
X["MultipleLines_OH"] = X["MultipleLines"].apply(lambda x: 1 if x == "No" else 0)
X["OnlineSecurity_OH"] = X["OnlineSecurity"].apply(lambda x: 1 if x == "Yes" else 0)
X["PaymentMethod_OH"] = X["PaymentMethod"].apply(
    lambda x: 1 if x == "Electronic check" else 0
)
standard_scaler.fit(
    X[
        [
            "tenure",
            "TotalCharges",
            "MultipleLines_OH",
            "OnlineSecurity_OH",
            "PaymentMethod_OH",
        ]
    ]
)


def transform(
    InternetService: str,
    Contract: str,
    tenure: int,
    TotalCharges: str,
    MultipleLines: str,
    OnlineSecurity: str,
    PaymentMethod: str,
) -> Dict[str, Union[int, float]]:
    if MultipleLines not in X["MultipleLines"].unique():
        raise ValueError("Invalid MultipleLines, got `{}`".format(MultipleLines))
    if OnlineSecurity not in X["OnlineSecurity"].unique():
        raise ValueError("Invalid OnlineSecurity, got `{}`".format(OnlineSecurity))
    if PaymentMethod not in X["PaymentMethod"].unique():
        raise ValueError("Invalid PaymentMethod, got `{}`".format(PaymentMethod))
    if isinstance(TotalCharges, str):
        TotalCharges = np.float64(TotalCharges.replace(" ", "0"))
    MultipleLines = 1 if MultipleLines == "No" else 0
    OnlineSecurity = 1 if OnlineSecurity == "Yes" else 0
    PaymentMethod = 1 if PaymentMethod == "Electronic check" else 0
    (
        tenure,
        TotalCharges,
        MultipleLines,
        OnlineSecurity,
        PaymentMethod,
    ) = standard_scaler.transform(
        [
            [
                tenure,
                TotalCharges,
                MultipleLines,
                OnlineSecurity,
                PaymentMethod,
            ]
        ]
    )[0]
    return {
        "InternetService_WoE": wt.single_transform(
            "InternetService", pd.Series([InternetService])
        )[0],
        "Contract_WoE": wt.single_transform("Contract", pd.Series([Contract]))[0],
        "tenure": tenure,
        "TotalCharges": TotalCharges,
        "MultipleLines_OH_No": MultipleLines,
        "OnlineSecurity_OH_Yes": OnlineSecurity,
        "PaymentMethod_OH_Electronic check": PaymentMethod,
    }


if __name__ == "__main__":
    print(transform("DSL", "One year", 3, 232345.22, "Yes", "No", "Electronic check"))
