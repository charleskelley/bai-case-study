"""
All the flow control flow code used to complete the Pfizer BAI Heart Disease
Case Study.
"""


# PROJECT SETUP ###############################################################

# Basics for interactive debugging work when necessary
# import inspect, os, sys

from typing import Union

# For Recursive Feature Elimination (RFE)
from numpy import mean
from numpy import std
from numpy.typing import ArrayLike
from pandas import DataFrame, concat
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# For data visualization
import dtreeviz
from matplotlib import pyplot

# Local utilities
from util.path import ProjectPath
from util.datadb import DataDB
from util.data import Splits, dataframe_profile_report

# Don't want to reload modules when working interactively
from IPython.core.getipython import get_ipython

# Instead of manualing entering in IPython session
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

# Easy to assets in project directory structure
paths = ProjectPath()

# Create a SQLite database connection to the datasets in `/data` directory
datadb = DataDB()
datadb.set_connection()

# Create a base patient DataFrame and remap column names to be more readable
patient_data = datadb.table_data("heart_disease_patient_data")
data_dictionary = datadb.table_data("heart_disease_patient_data_dictionary")
variable_alias_map = dict(zip(data_dictionary["variable"], data_dictionary["alias"]))
patient_data.rename(axis="columns", mapper=variable_alias_map, inplace=True)


# EXPLORATORY DATA ANALYSIS ###################################################

# ONLY NEED TO EXPORT DATA PROFILE TO HTML FILE ONCE
# Use browser to view `/output/patient-data-profile.html` file for reference
if not paths.output.joinpath("patient-data-profile.html").exists():
    dataframe_profile_report(
        patient_data,
        paths.output.joinpath("patient-data-profile.html"),
        correlations={
            "auto": {"calculate": True},
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
            "phi_k": {"calculate": True},
            "cramers": {"calculate": True},
        },
        title="Heart Disease Patient Data Profile",
    )


# SOLUTION DEVELOPMENT ########################################################

# OBJECTIVE: Select a classification model to predict heart disease diagnoses
# based on variables in the heart disease patient data set.

# Data cleaning and preparation is light for this dataset. There are no missing
# values and all data types are already numeric. The only things necessary are
# to standardize the numeric variables and one-hot encode the non-binary
# categorical variables


class PatientDataPrepped:
    """
    Class to encapsulate the data preparation steps for the heart disease data
    used in the modeling process.

    Args:
        data: DataFrame of heart disease patient data.

    Attributes:
        numeric_transformer: Pipeline to scale numeric features.
        categorical_transformer: Pipeline to one-hot encode categorical features.
        column_transformer: ColumnTransformer to apply numeric and categorical
            transformations to the data.
        features_raw: DataFrame of raw features.
        target_raw: DataFrame of raw target.
        features: Array of transformed features.
        feature_names: List of transformed feature names.
        target: Array of target feature values.
        target_name: Name of target feature.
    """

    # Numeric feature scaling
    NUMERIC_FEATURES_RAW = [
        "age",
        "exercise_st_depression",
        "rest_blood_pressure",
        "serum_cholesterol",
        "max_heart_rate",
    ]

    CATEGORICAL_FEATURES_RAW = [
        "chest_pain_type",
        "rest_ecg_results",
        "exercise_st_slope",
        "major_vessels_colored",
        "thalassemia",
    ]

    TARGET_FEATURE = "heart_disease_binary"

    def __init__(self, data: DataFrame):
        self.numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # non-binary categorical feature one-hot encoding
        # self.categorical_transformer = Pipeline(
        #     steps=[
        #         ("encoder", OneHotEncoder(sparse_output=False)),
        #         # ("selector", SelectPercentile(chi2, percentile=50)),
        #     ]
        # )

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("numeric", self.numeric_transformer, self.NUMERIC_FEATURES_RAW),
                # ("categorical", self.categorical_transformer, self.CATEGORICAL_FEATURES_RAW),
            ],
            remainder="passthrough",
        )

        self.features_raw = data.drop(columns=[self.TARGET_FEATURE], axis=1)
        self.target_raw = data[self.TARGET_FEATURE]

        self.features = self.column_transformer.fit_transform(
            self.features_raw, self.target_raw
        )
        self.feature_names = list(self.column_transformer.get_feature_names_out())

        # Shuffle of scalars is to enable upstream changes in
        self.target = self.target_raw
        self.target_name = self.TARGET_FEATURE


# Recursive Feature Elimination (RFE) and binary classification model


def candidate_pipelines(
    output_model: Union[DecisionTreeClassifier, LogisticRegression]
) -> dict[str, Pipeline]:
    """
    Create a dictionary of different core models to be used for performance
    comparison in the RFE feature selection process. Note that the final model
    is a decision tree classifier for all pipelines.

    Returns:
        Dictionary of model Pipeline objects.
    """
    models = dict()

    # logistic regression
    rfe = RFECV(estimator=LogisticRegression())
    model = output_model()
    models["logistic"] = Pipeline(steps=[("s", rfe), ("m", model)])

    # perceptron
    rfe = RFECV(estimator=Perceptron())
    model = output_model()
    models["perceptron"] = Pipeline(steps=[("s", rfe), ("m", model)])

    # classification and regression tree
    rfe = RFECV(estimator=DecisionTreeClassifier())
    model = output_model()
    models["cart"] = Pipeline(steps=[("s", rfe), ("m", model)])

    # random forest
    rfe = RFECV(estimator=RandomForestClassifier())
    model = output_model()
    models["forest"] = Pipeline(steps=[("s", rfe), ("m", model)])

    # gradient boost machine
    rfe = RFECV(estimator=GradientBoostingClassifier())
    model = output_model()
    models["gbm"] = Pipeline(steps=[("s", rfe), ("m", model)])

    return models


def evaluate_pipeline(data: PatientDataPrepped, pipeline: Pipeline) -> list[float]:
    """
    Use cross validation to evaluate a model.

    Args:
        pipeline: Model pipeline to evaluate.
        data: Prepped data to use for feature selection.

    Returns:
        Array of accuracy scores for the model on each cross validation fold.
    """
    cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(
        pipeline,
        data.features,
        data.target,
        scoring="accuracy",
        cv=cross_validation,
        n_jobs=-1,
    )

    return scores


def pipeline_accuracy(
    data: PatientDataPrepped, model_pipelines: dict[str, Pipeline]
) -> DataFrame:
    """
    Mean and standard deviation of model accuracy scores from cross validation.

    Args:
        data: Prepped data to use for feature selection.
        model_pipelines: Dictionary of model pipelines to evaluate.

    Returns:
        Dictionary of mean and standard deviation of model accuracy scores.
    """
    accuracy_scores = list()

    for name, pipeline in model_pipelines.items():
        scores = evaluate_pipeline(data, pipeline)
        accuracy = DataFrame(
            {
                "rfe_model": [name],
                "accuracy_mean": [mean(scores)],
                "accuracy_std": [std(scores)],
            }
        )
        accuracy_scores.append(accuracy)

    return concat(accuracy_scores)


def _clean_feature_names(feature_names: list[str]) -> list[str]:
    """
    Clean up the names of transformed features selected by RFE process.
    """
    return [
        name.replace("numeric__", "").replace("remainder__", "")
        for name in feature_names
    ]


def rfe_features_selected(
    rfe_model_name: str, pipeline: Pipeline, data: PatientDataPrepped
) -> DataFrame:
    """
    Use recursive feature elimination to select the best features for a model
    and show how they were selected and ranked.

    Args:
        rfe_model_name: Name of the model to use for feature selection.
        pipeline: Model pipeline to use for feature selection.
        data: Prepped data to use for feature selection.

    Returns:
        DataFrame of feature selection results for one pipeline.
    """
    feature_names = _clean_feature_names(data.feature_names)
    feature_selections = {
        "rfe_model": [rfe_model_name] * len(feature_names),
        "feature_name": feature_names,
    }

    rfe = pipeline.named_steps["s"]
    rfe.fit(data.features, data.target)

    selected, rank = [], []
    for i in range(len(feature_names)):
        selected.append(rfe.support_[i])
        rank.append(rfe.ranking_[i])

    feature_selections["selected"] = selected
    feature_selections["rank"] = rank

    return DataFrame(feature_selections)


def candidate_pipelines_features_selected(
    data: PatientDataPrepped, model_pipelines: dict[str, Pipeline]
) -> DataFrame:
    """
    Use recursive feature elimination to select the best features for each of
    the candidate pipelines and aggregate the results into a single DataFrame.

    Args:
        data: Prepped data to use for feature selection.

    Returns:
        DataFrame of feature selection results for all pipelines.
    """
    rfe_feature_selections = list()

    for name, pipeline in model_pipelines.items():
        features_selected = rfe_features_selected(name, pipeline, data)
        rfe_feature_selections.append(features_selected)

    return concat(rfe_feature_selections)


# ANALYSIS CONTROL FLOW #######################################################
# Everything above this point can be called without running the analysis or
# creating any new output artifacts

# Data prepped and ready for modeling
prepped_data = PatientDataPrepped(patient_data)


# DECISION TREE CLASSIFIER - candidate pipeline evaluation
pipelines_tree = candidate_pipelines(DecisionTreeClassifier)
pipelines_accuracy_tree = pipeline_accuracy(prepped_data, pipelines_tree)
# pipelines_accuracy_tree.to_csv(
#     paths.output.joinpath("pipelines-accuracy-tree.csv"), index=False
# )

pipelines_features_tree = candidate_pipelines_features_selected(
    prepped_data, pipelines_tree
)
# pipelines_features_tree.to_csv(paths.output.joinpath("pipelines-features-tree.csv"), index=False)


# LOGISTIC REGRESSION - candidate pipeline evaluation
pipelines_logistic = candidate_pipelines(LogisticRegression)
pipelines_accuracy_logistic = pipeline_accuracy(prepped_data, pipelines_logistic)
# pipelines_accuracy_logistic.to_csv(
#     paths.output.joinpath("pipelines-accuracy-logistic.csv"), index=False
# )

pipelines_features_logistic = candidate_pipelines_features_selected(
    prepped_data, pipelines_logistic
)
# pipelines_features_logistic.to_csv(paths.output.joinpath("pipelines-features-logistic.csv"), index=False)


# Unpacking the logistic regression pipeline to get the final model
full_logistic_pipeline = pipelines_logistic["logistic"]
full_logistic_pipeline.fit(prepped_data.features, prepped_data.target)
logistic_classifier = full_logistic_pipeline.named_steps["m"]
logistic_classifier_coefficients = list(logistic_classifier.coef_[0])
logistic_classifier_intercept = logistic_classifier.intercept_[0]

# Post RFE feature data
post_rfe_feature_data = prepped_data.features[:, full_logistic_pipeline["s"].support_]
post_rfe_feature_names = _clean_feature_names(
    [
        name
        for name, support in zip(
            prepped_data.feature_names, full_logistic_pipeline["s"].support_
        )
        if support
    ]
)

# Final logistic classifier coefficients
final_logistic_coefficients = list(
    zip(post_rfe_feature_names, logistic_classifier_coefficients)
)
final_logistic_coefficients.append(("intercept", logistic_classifier_intercept))

final_logistic_model = DataFrame(
    data=final_logistic_coefficients, columns=["feature", "coefficient"]
)
# final_logistic_model.to_csv(paths.output.joinpath("final-logistic-model.csv"), index=False)


# Playing with decision tree visualization
# viz_model = dtreeviz.model(
#     decision_classifier,
#     X_train=post_rfe_feature_data,
#     y_train=prepped_data.target,
#     feature_names=post_rfe_feature_names,
#     target_name="heart_disease_binary",
#     class_names=["True", "False"],
# )
# v = viz_model.view()
# v.show()
