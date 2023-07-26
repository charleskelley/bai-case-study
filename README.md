# Pfizer BAI Heart Disease Case Study

> This repository contains all the files created and work completed for the
> heart disease case study completed for the Pfizer data science role interview.

> **Note**
> The code used for the analysis is in the `main.py` file and the static
> artifacts produced by the analysis code are in the `output` directory, and
> the `resources` directory contains any static artifacts used for analysis
> presentation.

## Instructions From Pfizer

***Please put together your answers (slides) and any supporting code or
diagrams/visualizations or however else you feel would be the most appropriate
way to express your answers. Please e-mail this to us.***

## Case Study

We are interested in understanding what are the main contributing factors
toward heart disease?

**Tasks:**

<ol type="a">
 <li>
   Use a predictive model to identify the main contributing factors towards heart disease
 </li>
 <li>
   Put together a mock-up visual to share the results of your analysis. Be prepared to discuss which model(s) you used and why
 </li>
</ol>

### Case Study Data

We have been given a patient-level dataset that measures key demographic and
health outcomes for a given patient.

This dataset has been flagged whether a patient was ultimately diagnosed with
heart disease in the target column. We have also been provided a data dictionary
in the Heart_Disease_Patient_Data_Dictionary tab.

**Dataset**

Data for the case study analysis is sourced from the
`Heart_Disease_Patient_Data` worksheet in the `bai-case-study-heart.xlsx` Excel
workbook and saved to the `heart-disease-patient-data` CSV file in the `data`
directory. In addition to the heart disease patient data, a hidden worksheet
with data tables was also found in the workbook. Those data tables were saved to
individual CSV files in the `data` directory for reference.

**Data Dictionary**

Additionally, below is the data dictionary sourced from the
`Heart_Disease_Patient_Data_Dictionary` worksheet in the previously mentioned
Excel workbook modified with an additional column added to give variable names
an alias that is more human-readable and descriptive.

> **Note**
> From here forward, the variable names will be referred to by their alias

<br>

| variable	     | alias                     | description                                                    |
|:--------------|:--------------------------|:---------------------------------------------------------------|
| `age`         | `age`                     | age in years                                                   |
| `sex`         | `male_binary`             | (1 = male; 0 = female)                                         |
| `cp`          | `chest pain type`         | chest pain type                                                |
| `trestbps`    | `rest_blood_pressure`     | resting blood pressure (in mm Hg on admission to the hospital) |
| `chol`        | `serum_cholestoral`       | serum cholestoral in mg/dl                                     |
| `fbs`         | `fast_bg_above120_binary` | (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)        |
| `restecg`     | `rest_ecg_results`        | resting electrocardiographic results                           |
| `thalach`     | `max_heart_rate`          | maximum heart rate achieved                                    |
| `exang`       | `execise_angina_binary`   | exercise induced angina (1 = yes; 0 = no)                      |
| `oldpeak`     | `exercise_st_depression`  | ST depression induced by exercise relative to rest             |
| `slope`       | `exercise_st_slope`       | the slope of the peak exercise ST segment                      |
| `ca`          | `major_vessels_colored`   | number of major vessels (0-3) colored by flourosopy            |
| `thal`        | `thalessemia`             | 3 = normal; 2 = fixed defect; 1 = reversable defect            |
| `target`      | `heart_disease_binary`    | Flag if the patient has heart disease (1) or not (0)           |

> The data heart disease patient dataset appears to be a standard dataset of
> variables used in assessing heart disease. See the [Background Research](#step-2-background-research)
> section below for more information on a paper that uses the same variables to
> classify heart disease.

## Heart Disease Contributing Factors analysis

### Step 1: Exploratory Data Analysis

To see what the data looks like, I ran a Pandas Profiling report on the heart
disease patient data. The HTML full data profile report is saved to
[`/output/patient-data-profile.html`](output/patient-data-profile.html) and can
be easily viewed by opening the file in a web browser.

#### Notable Variable Characteristics

* `male_binary` - has a high imbalance toward male sex with 207 observations
  indicated as male and 96 as not-male (female)
* `exercise_st_depression` - has an extreme positive skew with most values being
  `0.0` but with range of `0.0 - 6.2` and a skewness statistic of `~1.27` and
  kurtosis or `~1.57`
* All numeric variables aside from `exercise_st_depression` have a normal
  distribution with and any moderate positive or negative skewness is due to
  one or a few outlier observations

### Step 2: Problem Framing 

#### Objective 

Given the data provided, the objective is to identify the main contributing
factors that lead to a heart disease diagnosis.

To assess the relative contribution of multiple factors, a simple linear
classification model can be built where the target variable is
`heart_disease_binary` variable and all other variables are assessed as
predictors.

The assumption that their parametric contributions to the target variable are
additive and can be used as a proxy for their contribution to the heart disease
diagnosis.

#### Key Considerations

There are many different binary classification models that could be used to try
and get the best performing model on strictly the classification task. However,
the request was for contribution analysis. Therefore, the model should be
selected with explainability as a major consideration.

With the the 24-48 time constraint and explainability as chief criteria. A
linear classification model with dimensionality reduction is an obvious option.

### Step 3: Background Research

#### Heart Disease

I did some cursory research on heart disease and found the following paper that
uses the variables available in the patient data to classify heart disease

* [Heart disA Hybrid Classification System for Heart Disease Diagnosis Based on the RFRS Method](https://www.hindawi.com/journals/cmmm/2017/8272091/)

#### Dimensionality Reduction

* [Comparison of PCA and RFE-RF Algorithm in Bankruptcy Prediction](https://dergipark.org.tr/en/download/article-file/2272320#:~:text=The%20most%20important%20difference%20was,features%20into%20a%20lower%20dimension)

