# Credit-risk-prediction
## Team Member
| **Student Name** | **Student ID** |
| ------------- | ------------- |
| Pan Kangyu | 2301212370 |
| Zhang Qiuyan | 2301212290 |  

## Project Introduction
### 1. Project Purpose
This is an ongoing [Kaggle Competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) where the task is to predict the default risk of different cases (a binary classification task) using some internal and external data including credit history, tax information, deposits, etc.

### 2. Overall Framework
* Exploratory data analysis.
* Preprocess data, including missing value filling, category variable encoding, continuous variable binning, standardization, etc.
* Divide features into 3 groups according to their meanings.
* Build the LightGBM model to filter important features and make prediction.
* Use Logistic Regression to ensemble the predicted scores(probabilities) of the three models.

## 1. Exploratory Data Analysis
### 1.1 Data Structure
**Input**: 465 features from both internal and external data resources.<br />
<img width="575" alt="feature tables" src="https://github.com/zsevenn/Credit-risk-prediction/assets/95988969/d8e8967a-ada2-4c65-b4bc-47bbb61ab176">

**Output**: "target" variable, taget = 1 means that the client has defaulted, tagert = 0 means that the client has not defaulted.<br />

**More details**
* Base table<br />
Base table stores the basic information, mainly including case_id(unique code for each client) and target(valiable indicating whether client will default). "case_id"  is a unique identification of every observation and we use it to join the other tables containing feature values to the base table.<br />
* Depth values<br />
  * depth=0 - These are static features directly tied to a specific case_id.<br />
  * depth=1 - Each case_id has an associated historical record, indexed by num_group1.<br />
  * depth=2 - Each case_id has an associated historical record, indexed by both num_group1 and num_group2.<br />
* Feature Naming<br />
Various predictors are transformed, so to have the following notation for similar groups of transformations: P M A D T L.<br />
  * P - Transform DPD (Days past due)
  * M - Masking categories
  * A - Transform amount
  * D - Transform date
  * T - Unspecified Transform
  * L - Unspecified Transform
    
For example, "actualdpd_943P" is a feature meaning that Days Past Due (DPD) of previous contract (actual).

### 1.2 Descriptive Statistics 
* Imbalanced dataset<br />
The dataset is extremely imbalanced, which is expectable for such domain.<br />
![imbalance dataset](https://github.com/zsevenn/Credit-risk-prediction/assets/95988969/3921d89d-5f90-4336-8f5b-d52a24567207)


* Data distribution based on the dates of making decisions regarding loan approval<br />
![distribution](https://github.com/zsevenn/Credit-risk-prediction/assets/95988969/dcd4f415-7be4-42b0-9e81-49c139753ed9)
![distribution weekday](https://github.com/zsevenn/Credit-risk-prediction/assets/95988969/ae405a3f-d325-43fa-995b-28d044d3639d)
![target mean](https://github.com/zsevenn/Credit-risk-prediction/assets/95988969/41efda10-242c-4b44-8f9c-8251e0feb936)

From the above three graphs, the overall proportion of client defaults is low, and according to monthly statistics, this proportion has decreased since April 2020. And interestingly, the bank makes decision even on the weekends. Clearly, this process is automated and runs on a predefined schedule.

## 2. Data Preprocessing

### 2.1 Characteristics of Datasets
  * A large number of non-continuous features(including date-type and categorical features).
  * A large number of missing values.
  * A large number of features of different levels (depth=1, depth=2).
### 2.2 Procedure of Preprocessing:
  1. Date variable: directly converted to the number of days from the `desion_date`, and then we will treat them as continuous variables.
  2. Continuous variables: 
      1) If the proportion of missing values is less than the threshold (like 0.3), fill in the mean or median.
      2) If the proportion is greater than the threshold, continuous variables will be binned (generally 20 bins), and missing values are divided into a separate box, and then we will use `WOE`(Weight of Evidence Encoding) to convert the binned data into numeric values. 
      The main reason is that we think that **a large proportion of missing values in a feature may contain information relevant to credit default risk**, e.g. a certain type of loan history may be missing for a certain group of people.
  2. Discrete variables: 
      1) If the proportion of missing values is less than the threshold (like 0.3), fill with the mode.
      2) If the proportion is greater than the threshold, the missing values will be divided into a separate category.
      3) Encode all discrete features with `WOE` method. 
  3. Standardization.
  4. Reduce memory consumption.

**About WOE:**     
`WOE` is a supervised encoding method for binary classification problems, by computing the weight of evidence for each category to represent its relationship with the target variable. This method is widely used in the field of credit risk assessment.
 
$$
WOE_i = ln(\frac{Bad_i}{Bad_T}/\frac{Good_i}{Good_T})
$$
 
$Good_i$ denote the number of `target = 0` (e.g., customers who have not defaulted on their loans) in the category and $Good_T$ denote the total number of `target = 0`.
$Bad_i$ denote the number of `target = 1` (e.g., customers who have defaulted on their loans) in the category and $Bad_T$ denote the total number of `target = 1`.
 
 Advantages of WOE Encoding:
   1. No increase in data dimensionality (compared to One-hot encoding).
   2. The transformed values are meaningful (compared to Ordinary encoding), and when combined with binning, allow us to capture the information represented by the missing values.

**Advantages of binning:**    
    * Help capture nonlinear relationships(combined with WOE Encoding).
    * Help treat missing values of a continuous variable as a single category.
    * Ignore small differences in some features.
    
### 2.3 Procedure of Aggregating
#### Level=1
Each `case_id` has an associated historical record, indexed by num_group1. It means that for one `case_id`, there will be multiple values for a particular feature because there are multiple `num_group1` for the case, and each case has a different number of `num_group1`.

Aggregation methods:
1. If the column is a numeric value, generate the mean, maximum, minimum and number of non-null values among each `num_group1` under the same `case_id`.
2. If the column is a categorical value, generate pluralities and the number of non-null values among each `num_group1` under the same `case_id`.
#### Level=2
Each `case_id` has an associated historical record, indexed by both num_group1 and num_group2. In addition to `num_group1`, there will also be multiple values for a particular feature in the same `num_group1` under the same `case_id`, because there will also be multiple `num_group2` under each `num_group1` for each `case_id`.

Aggregation methods:
1. If the column is a numeric value, generate the mean, maximum, minimum and number of non-null values among each `num_group2` under the same `case_id`.
2. If the column is a categorical value, generate pluralities and the number of non-null values among each `num_group2` under the same `case_id`.
3. To better capture the relationship between `num_group1` and `num_group2`, we also compute the mean and maximum of the number of non-nulls in different `num_group1` under the same case_id.

## 3. Model
### 3.1 Feature Grouping
The number of original features in the dataset is more than 400, and after aggregating and preprocessing, the total number of features is more than 1,000. Considering the efficiency of model training, we categorize the features into three groups according to their meanings, meaning that we will train three LigntGBM models on three parts of features respectively.

- The first group of data: previous loan applications and credit history (344 features).
- The second group of data: records related to contracts and mortgages (523 features).
- The third group of data: personal information (basic information, various deposits and tax information) (178 features).

```
merging_dict['group1'] = ["applprev_1", "applprev_2", "static_0", "static_cb_0"]
merging_dict['group2'] = ["credit_bureau_a_1", "credit_bureau_b_1",
                          "credit_bureau_a_2", "credit_bureau_b_2"]
merging_dict['group3'] = ["person_1", "person_2", 
                          "other_1", "deposit_1", "debitcard_1", 
                          "tax_registry_a_1", "tax_registry_b_1", "tax_registry_c_1"]   
```
**Difference from modular approach:**    
Because we have filled in all the missing values, i.e., we have utilized the missing values as a form of information, so that each sample appears in all three models.

### 3.2 Local Split
Since the test datasets provided by the competition do not contain `target`, we need to split a 1/10 validation set in which the data does not participate in any model training process, so that it is can be used to validate the effectiveness of the final model locally.

### 3.3 LightGBM
#### 3.3.1 Process of Model Training:
1. Train model with all features initially.
2. Tune hyperparameters of the model with all features.
2. Select important features using the best model.
3. Retrain the best model with selected features. 
4. Test the model with local test data.    

In cross validation, we use `stratifiedGroupKFold`, which combines `StratifiedKFold` and `GroupKFold`.
- `GroupKFold` make sure that samples from the same week won't be divided into different folds.
- `StratifiedKFold` make sure that the class distributions of the training sets and validation sets are consistent with the original dataset.
#### 3.3.2 Hyperparameter Tuning:
```
import optuna

def objective(trial, X, y):
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 500, 1000, 5000]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.1, 0.15, 0.2]),
        
        "max_depth": trial.suggest_int("max_depth", 5, 20, step=5),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1000, 9000, step=2000),
        
        "lambda_l1": trial.suggest_int("lambda_l1", 20, 100, step=20),
        "lambda_l2": trial.suggest_int("lambda_l2", 20, 100, step=20),   
        
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 20, step=5),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.6, step=0.2),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.6, step=0.2),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8, step=0.2),
        "colsample_bynode": 0.8,
        
        "random_state": 2024,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "verbose": -1
    }
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False) 

    cv_scores = np.empty(5)
    for cv_idx, (train_index, valid_index) in enumerate(cv.split(X, y, groups=weeks)):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]
        
        model = lgb.LGBMClassifier(**param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[( X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(100)
            ],
        )
        
        preds = model.predict_proba(X_valid)[:,1]
        cv_scores[cv_idx] = roc_auc_score(y_valid, preds)

    return np.mean(cv_scores)
study = optuna.create_study(direction="maximize", study_name="LGBM Classifier")
func = lambda trial: objective(trial, X, y)
study.optimize(func, n_trials=10)
```
#### 3.3.3 Model Performance of Each Group
##### Group 1
- Selected features:    
We selected 203 features from 344 features in Group 1, which has the importance values higher than 20.
![output_group1](https://github.com/zsevenn/Credit-risk-prediction/assets/147133482/848b43a4-6d68-42ce-beaf-011f166bbc15)     
    
- AUC score on local test set: 0.83788
##### Group 2
- Selected features:    
We selected 81 features from 530 features in Group 2, which has the importance values higher than 10.
![output_group2](https://github.com/zsevenn/Credit-risk-prediction/assets/147133482/aa397a8f-177d-480a-8a5d-4c88ea014347)
    
- AUC score on local test set: 0.82344
##### Group 3
- Selected features:    
We selected 45 features from 189 features in Group 3, which has the importance values higher than 5.
![output_group3](https://github.com/zsevenn/Credit-risk-prediction/assets/147133482/f40393bb-7a41-4c09-bfa4-a65fb718b46a)

- AUC score on local test set: 0.75745

### 3.4 Logistic Regression
We use Logistic Regression to ensemble the scores (probability) from three models.

```
param_grid = {
    'C': [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'penalty': ['l2'],
    'fit_intercept': [False],
    'class_weight': ['balanced'],
    'solver': ['sag'],
    'max_iter': [200, 400, 600, 800]
}

log_reg = LogisticRegression()
grid_search = GridSearchCV(log_reg, param_grid, cv=10, scoring='roc_auc')
grid_search.fit(X_LR, y_LR)
```
- AUC score on local test set: 0.88854
- ROC curve:    
![output)final](https://github.com/zsevenn/Credit-risk-prediction/assets/147133482/76730b76-59a4-4151-9bf0-6eccff7d17ae)

