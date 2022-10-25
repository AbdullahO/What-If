import os

import pandas as pd
import pytest

from algorithms.snn import SNN

current_dir = os.getcwd()


@pytest.fixture(scope="session")
def snn_expected_query_output() -> pd.DataFrame:
    query_output = pd.read_pickle(
        os.path.join(current_dir, "data/stores_sales_simple/snn_query_output.pkl")
    )
    return query_output


@pytest.fixture(scope="session")
def snn_test_df() -> pd.DataFrame:
    _snn_test_df = pd.read_csv(
        os.path.join(current_dir, "data/stores_sales_simple/stores_sales_simple.csv")
    )
    return _snn_test_df


def test_snn(snn_test_df, snn_expected_query_output):
    model = SNN(verbose=False)
    model.fit(
        df=snn_test_df,
        unit_column="unit_id",
        time_column="time",
        metrics=["sales"],
        actions=["ads"],
    )
    model_query_output = model.query(
        [0],
        ["2020-01-10", " 2020-02-19"],
        "sales",
        "ad 0",
        ["2020-01-10", " 2020-02-19"],
    )

    assert snn_expected_query_output.equals(
        model_query_output
    ), "Query output difference"
    assert model.actions_dict == {
        "ad 2": 0,
        "ad 0": 1,
        "ad 1": 2,
    }, "Actions dict difference"
