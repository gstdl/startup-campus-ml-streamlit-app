from typing import Dict, Union
import pandas as pd
from woe_transformer import WoeTransformer

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
wt = WoeTransformer(X, y, {"No": "Events", "Yes": "Non events"})
wt.single_fit("InternetService")
wt.single_fit("Contract")


def transform(
    InternetService: str,
    Contract: str,
    tenure: int,
    TotalCharges: float,
    MultipleLines: str,
    OnlineSecurity: str,
    PaymentMethod: str,
) -> Dict[str, Union[int, float]]:
    if MultipleLines not in X["MultipleLines"].unique():
        raise ValueError("Invalid MultipleLInes")
    if OnlineSecurity not in X["OnlineSecurity"].unique():
        raise ValueError("Invalid OnlineSecurity")
    if PaymentMethod not in X["PaymentMethod"].unique():
        raise ValueError("Invalid PaymentMethod")
    return {
        "InternetService_WoE": wt.single_transform(
            "InternetService", pd.Series([InternetService])
        )[0],
        "Contract_WoE": wt.single_transform("Contract", pd.Series([Contract]))[0],
        "tenure": tenure,
        "TotalCharges": TotalCharges,
        "MultipleLines_OH_No": 1 if MultipleLines == "No" else 0,
        "OnlineSecurity_OH_Yes": 1 if OnlineSecurity == "Yes" else 0,
        "PaymentMethod_OH_Electronic check": 1
        if PaymentMethod == "Electronic check"
        else 0,
    }


if __name__ == "__main__":
    print(transform("DSL", "One year", 3, 232345.22, "Yes", "No", "random"))