import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulation.patient_management.priority import PriorityCalculator


@pytest.fixture()
def df():
    """Fixture that returns a sample DataFrame with missing required columns."""
    return pd.DataFrame(
        columns=["setting", "priority", "days waited"],
        data=[["Inpatient", "Urgent", 10], ["Outpatient", "Routine", 30]],
    )


@pytest.fixture()
def pc():
    return PriorityCalculator(
        [
            "A&E patients",
            "inpatients",
            "Breach",
            "Days waited",
            "Over minimum wait time",
            "Under maximum wait time",
        ],
    )


@pytest.fixture()
def reg_map():
    config_file = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "simulation/config/min_max_wait_mapping.json"
    )
    with open(config_file, "r") as fin:
        return json.load(fin)


def test_sorted_indices_assertion(df, pc):
    """Test to ensure that PriorityCalculator raises an AssertionError if required columns are missing."""
    incomplete_df = df.drop(columns=["days waited"])

    with pytest.raises(AssertionError):
        pc.calculate_sorted_indices(incomplete_df)


def test_apply_regex_map_raise(pc, reg_map):
    """Test to ensure that if a new value that is not matched by the regex is present an error is raised."""
    with pytest.raises(ValueError):
        pc.apply_regex_map("Biscuits!", reg_map, default_value=[0, 0])


def test_apply_regex_map(pc, reg_map):
    """Test to ensure various regex matches return correct values"""
    assert pc.apply_regex_map("2 Week wait", reg_map, default_value=[0, 0]) == [7, 14]
    assert pc.apply_regex_map("Routine", reg_map, default_value=[0, 0]) == [21, 126]
    assert pc.apply_regex_map("", reg_map, default_value=[0, 0]) == [0, 0]


def test_calculate_min_and_max_wait_times(df, pc):
    """Test to check that 2 new columns are set up: MinWaitTime, MaxWaitTime"""
    min_and_max_wait_times = pc.calculate_min_and_max_wait_times(df)

    assert (min_and_max_wait_times == np.array([[7, 14], [21, 126]])).all()


def test_calculate_sorted_indices(pc, df):
    """Test to check the priority orders for various scenarios."""
    pc.priority_order = ["inpatients", "Days waited"]
    assert pc.calculate_sorted_indices(df).tolist() == [
        0,
        1,
    ]

    pc.priority_order = ["Days waited", "inpatients"]
    assert pc.calculate_sorted_indices(df).tolist() == [
        1,
        0,
    ]

    pc.priority_order = ["Under maximum wait time"]
    assert pc.calculate_sorted_indices(df).tolist() == [0, 1]


if __name__ == "__main__":
    pytest.main()
