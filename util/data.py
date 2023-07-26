"""
Classes and functions for working with data workflows
"""
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, Union

from numpy.typing import ArrayLike
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport

from util.path import ProjectPath


def get_dataframe(dataset_name: str) -> DataFrame:
    """
    Get a pandas DataFrame for a dataset in a subdirectory within the project's
    data directory.

    Args:
        dataset_name:  Name of a subdirectory within the project's data directory
            containing a CSV file with the same name as the subdirectory.

    Returns:
        Pandas DataFrame for the given dataset.
    """
    paths = ProjectPath()
    dataset_path = getattr(paths, dataset_name)
    dataset_csv = dataset_path.joinpath(f"{dataset_name}.csv")

    return read_csv(dataset_csv)


def dataframe_profile_report(
    dataframe: DataFrame,
    report_path: Union[str, PathLike] = "data-profile-report.html",
    **kwargs: Any,
) -> None:
    """
    Generate a data profiling report for a given pandas DataFrame and save it

    Args:
        dataframe: Pandas DataFrame to profile.

        report_path: Path to save the report HTML file to. The file will be
            saved to the working directory if only a name is provided and
            `.html` will be appended to the name if not provided.

        **kwargs: Keyword arguments to pass to pandas_profiling.ProfileReport
    """
    if isinstance(report_path, str):
        try:
            Path(report_path).parent.resolve(strict=True)
        except FileNotFoundError:
            report_path = Path(report_path)

    if not report_path.suffix:
        report_path = report_path.with_suffix(".html")

    profile = ProfileReport(dataframe, **kwargs)
    profile.to_file(report_path)


@dataclass
class DataSplit:
    """
    Container for holding feature and target data splits.
    """
    features: Union[DataFrame, ArrayLike]
    target: Union[DataFrame, ArrayLike]


class Splits:
    """
    Make keeping track of test and train data splits easier for a given dataset.

    Args:
        X (ArrayLike): Features ndarray or a DataFrame with only feature columns
            from the dataset to be split for training and testing.

        y (ArrayLike): Target ndarray or a DataFrame with only target column
            from the dataset to be split for training and testing.

        random_state (int, optional): Random state for reproducibility.
            Defaults to 1.

        **kwargs: Keyword arguments to pass to sklearn.model_selection.train_test_split

    Attributes:
        train (NamedTuple): NamedTuple with X and y attributes for training data.

        test (NamedTuple): NamedTuple with X and y attributes for testing data.
    """

    def __init__(
        self, features: ArrayLike, target: ArrayLike, random_state: int = 1, **kwargs
    ):
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, random_state=random_state, **kwargs
        )

        self.train = DataSplit(train_features, train_target)
        self.test = DataSplit(test_features, test_target)
