"""
All the flow control flow code used to complete the Pfizer BAI Heart Disease
Case Study.
"""


# PROJECT SETUP ###############################################################

# Basics for interactive work
import inspect, os, sys

# Data wrangling
import pandas as pd

# Statistical analysis
import sklearn as sk

# Data visualization
import plotly.express as px

# Utilities
from util.path import ProjectPath
from util.datadb import DataDB
from util.data import Splits, dataframe_profile_report

# Don't want to reload modules when working interactively
from IPython.core.getipython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Easy to assets in project directory structure
paths = ProjectPath()

# Create a SQLite database connection to the datasets in `/data` directory
datadb = DataDB()
datadb.set_connection()

# Create a base patient DataFrame and remap column names to be more readable
patient_data = datadb.table_data("heart_disease_patient_data")
data_dictionary = datadb.table_data("heart_disease_patient_data_dictionary")
variable_alias_map = dict(zip(data_dictionary["variable"], data_dictionary["alias"]))

# Base patient data DataFrame
patient_data.rename(axis="columns", mapper=variable_alias_map, inplace=True)

# Patient data DataFrame splits for training and testing ML models
patient_data_splits = Splits(
    patient_data.drop(columns=["heart_disease_binary"]),
    patient_data["heart_disease_binary"]
)


# EXPLORATORY DATA ANALYSIS ###################################################

# ONLY NEED TO EXPORT DATA PROFILE TO HTML FILE ONCE
# Use browser to view `/output/patient-data-profile.html` file for reference
# dataframe_profile_report(
#     patient_data,
#     paths.output.joinpath("patient-data-profile.html"),
#     correlations={
#         "auto": {"calculate": True},
#         "pearson": {"calculate": True},
#         "spearman": {"calculate": True},
#         "kendall": {"calculate": True},
#         "phi_k": {"calculate": True},
#         "cramers": {"calculate": True},
#     },
#     title="Heart Disease Patient Data Profile"
# )


# SOLUTION DEVELOPMENT ########################################################

# OBJECTIVE: Build a classification model to predict heart disease diagnoses
# based on variables in the heart disease patient data set.

# DATA CLEANING AND PREPARATION IS NOT REQUIRED FOR THIS DATA SET. There are no
# missing values and all data types are already numeric. Any normalization or
# standardization of the data can be done as part of the model development
# process.

# 1. MODEL SELECTION

# 1. Variable Selection -