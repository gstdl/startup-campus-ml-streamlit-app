import streamlit as st
import numpy as np
import pickle
import lime.lime_tabular
from .dataloader import transform, X
import concurrent.futures

THRESHOLD = 0.44


with open("data/model/LogisticRegression.pkl", "rb") as f:
    model = pickle.load(f)


def predict(
    InternetService: str,
    Contract: str,
    tenure: int,
    TotalCharges: float,
    MultipleLines: str,
    OnlineSecurity: str,
    PaymentMethod: str,
) -> str:
    features = transform(
        InternetService,
        Contract,
        tenure,
        TotalCharges,
        MultipleLines,
        OnlineSecurity,
        PaymentMethod,
    )

    input_array = np.array(list(features.values())).reshape(1, -1)
    pred_proba = model.predict_proba(input_array)
    pred = np.where(pred_proba[:, 0] > THRESHOLD, "No", "Yes")[0]

    return pred_proba[0][0], pred


@st.cache
def get_explainer():
    from .dataloader import X, y, transform, selected_features
    # from streamlit.scriptrunner import get_script_run_ctx

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # ctx = get_script_run_ctx()
        for ix, row in X.iterrows():
            futures.append(
                (
                    ix,
                    executor.submit(
                        transform,
                        row["InternetService"],
                        row["Contract"],
                        row["tenure"],
                        row["TotalCharges"],
                        row["MultipleLines"],
                        row["OnlineSecurity"],
                        row["PaymentMethod"],
                        # ctx=ctx,
                    ),
                )
            )

    for f in futures:
        ix = f[0]
        temp = f[1].result()
        row = list(temp.values())
        columns = temp.keys()
        X.loc[ix] = row

    X.columns = list(columns)
    print(X.head())

    y = y.map({"No": 1, "Yes": 0})
    # categorical_names = [
    #     "MultipleLines_OH_No",
    #     "OnlineSecurity_OH_Yes",
    #     "PaymentMethod_OH_Electronic check",
    # ]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=selected_features,
        class_names=["Yes", "No"],
    )
    return explainer


if __name__ == "__main__":
    print(predict("DSL", "One year", 3, 232345.22, "Yes", "No", "Mailed check"))
