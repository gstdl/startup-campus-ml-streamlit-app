import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from tqdm import tqdm

from lib.dataloader import X, y
from lib.machine_learning import predict

import warnings

warnings.filterwarnings("ignore")

st.header("Business Insights")
st.write("PLACEHOLDER")

st.header("Model Insights")


@st.cache
def do_calculation():
    pred_proba = np.zeros(len(X), dtype=np.float64)
    pred = []
    for ix, row in tqdm(X.iterrows()):
        temp = predict(
            row["InternetService"],
            row["Contract"],
            row["tenure"],
            row["TotalCharges"],
            row["MultipleLines"],
            row["OnlineSecurity"],
            row["PaymentMethod"],
        )
        pred_proba[ix] = temp[0]
        pred.append(temp[1])
    return pred_proba, pred


with st.spinner("Crunching numbers..."):
    pred_proba, pred = do_calculation()

# @st.cache
def plot_confusion_matrix() -> plt.figure:
    print(pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        pd.DataFrame(confusion_matrix(y, pred), ["No", "Yes"], ["No", "Yes"]),
        annot=True,
        fmt=".0f",
        cbar=False,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Prediction")
    ax.set_title("Confusion Matrix")
    return fig


st.pyplot(plot_confusion_matrix())

# @st.cache
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y.map({"No": 1, "Yes": 0}), pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.set_title("Receiver Operating Characteristic")
    ax.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    return fig


st.pyplot(plot_roc_curve())

# @st.cache
def plot_precision_recall_curve() -> plt.figure:
    precision, recall, thresholds = precision_recall_curve(
        y.map({"No": 1, "Yes": 0}), pred_proba
    )
    thresholds = np.append(thresholds, 1)
    fig, ax = plt.subplots()
    ax.step(recall, precision, color="b", alpha=0.4, where="post")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("Precision-Recall curve")
    return fig


st.pyplot(plot_precision_recall_curve())
