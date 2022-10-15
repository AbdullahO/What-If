import pandas as pd
import pytest

import os
import sys
current_dir = os.getcwd() # whatIf folder
curr_algos_path = os.path.join(current_dir, "current_algorithms/si")
sys.path.append(curr_algos_path)
from current_algorithms.si import SI as current_SI

# import algorithms.si as si

@pytest.fixture(scope="session")
def pre_df() -> pd.DataFrame:
    pre_df = pd.read_pickle(os.path.join(current_dir, "data/basque_pre_df.pkl"))
    return pre_df


@pytest.fixture(scope="session")
def post_df() -> pd.DataFrame:
    post_df = pd.read_pickle(os.path.join(current_dir, "data/basque_post_df.pkl"))
    return post_df


def test_current_synthetic_interventions(pre_df, post_df):
    basque = current_SI.SI(t=0.99, center=False)
    basque.fit(pre_df, post_df)
    import ipdb; ipdb.set_trace()
    # TODO: assert something
    

