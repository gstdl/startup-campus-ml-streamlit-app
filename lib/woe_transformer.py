from typing import Dict, Any, Callable, Union, Tuple
import pandas as pd
import numpy as np


class WoeTransformer:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_mapper: Dict[Any, str],
    ) -> None:
        # validate y_mapper
        if len(y_mapper) != y.nunique():
            raise ValueError("Invalid `y_mapper`")
        if "Non events" not in y_mapper.values() or "Events" not in y_mapper.values():
            raise ValueError("Invalid `y_mapper`")
        self.X: pd.DataFrame = X
        self.y: pd.Series = y.map(y_mapper)
        self.column_mapper: Dict[str, Callable[[pd.Series], pd.Series]] = {}

    # def fit(
    #     self, mapper_functions: Dict[str, Callable[[pd.Series], pd.Series]] = None
    # ) -> None:
    #     if mapper_functions is None:
    #         mapper_functions = {}
    #     for column in self.X.columns:
    #         mf = mapper_functions.get(column, lambda x: x)
    #         self.single_fit(column, mf)

    # def transform(self) -> pd.DataFrame:
    #     df = pd.DataFrame()
    #     for column in self.X.columns:
    #         df[f"{column}_WoE"] = self.single_transform(column).values
    #     return df

    # def fit_transform(
    #     self, mapper_functions: Dict[str, Callable[[pd.Series], pd.Series]] = None
    # ) -> pd.DataFrame:
    #     self.fit(mapper_functions)
    #     return self.transform()

    def single_fit(
        self,
        column: str,
        mapper_function: Callable[[pd.Series], pd.Series] = lambda x: x,
        return_df_and_iv: bool = False,
    ) -> Union[Tuple[pd.DataFrame, float], None]:
        self.column_mapper[column] = mapper_function

        x = self.column_mapper[column](self.X[column])
        df_woe = pd.crosstab(x, self.y)
        df_woe = df_woe[["Non events", "Events"]]
        df_woe["% of Observations"] = (df_woe["Non events"] + df_woe["Events"]) / (
            df_woe["Non events"] + df_woe["Events"]
        ).sum()
        df_woe["% of Non events"] = df_woe["Non events"] / df_woe["Non events"].sum()
        df_woe["% of Events"] = df_woe["Events"] / df_woe["Events"].sum()
        df_woe["WoE"] = np.log(
            (np.maximum(df_woe["Non events"], 0.5) / df_woe["Non events"].sum())
            / (np.maximum(df_woe["Events"], 0.5) / df_woe["Events"].sum())
        )
        df_woe["IV"] = (df_woe["% of Non events"] - df_woe["% of Events"]) * df_woe[
            "WoE"
        ]
        if return_df_and_iv:
            return df_woe, df_woe["IV"].sum()

    def single_transform(
        self,
        column: str,
        x: str,
    ) -> pd.Series:
        df_woe, _ = self.single_fit(column, self.column_mapper[column], True)
        # x_transformed = self.X[column].copy()
        # x_transformed = self.column_mapper[column](x_transformed)
        # x_transformed = x_transformed.apply(lambda x: df_woe.loc[x, "WoE"])

        x_transformed = self.column_mapper[column](x)
        x_transformed = x_transformed.apply(lambda x: df_woe.loc[x, "WoE"])

        return x_transformed

    # def single_fit_transform(
    #     self,
    #     column: str,
    #     mapper_function: Callable[[pd.Series], pd.Series] = lambda x: x,
    # ) -> pd.Series:
    #     self.fit(column, mapper_function)
    #     return self.transform(column)
