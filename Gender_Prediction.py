import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, classification_report


# FUNCTION
def evaluation(model, Xtest, ytest, y_pred):
    ytest_pred = model.predict(Xtest)
    print("The f1 score is " + str(np.round(f1_score(ytest, ytest_pred, average='weighted'), 2)))
    #plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='pred')
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = model.classes_)
    disp.plot()
    plt.show()
    plt.title("confusion matrix")
    plt.show()
    print("The recall score is " + str(np.round(recall_score(y_test, y_pred), 2)))
    print("The precision score is " + str(np.round(precision_score(y_test, y_pred), 2)))

def plot_top_feature_importance(model, X_train, top=10):
    plt.rcParams.update({'font.size': 14})
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    top_features = sorted_indices[0:top]
    plt.title('Feature Importance')
    plt.bar(range(top), importances[top_features], align='center')
    plt.xticks(range(top), X_train.columns[top_features], rotation=90)
    plt.tight_layout()
    plt.show()

def plot_categorical_distribution(data, col_name):
    plt.figure(figsize=(20, 8))
    plot = sns.countplot(x=data)
    total = len(data)
    for patch in plot.patches:
        percentage = '{:.1f}%'.format(100 * (patch.get_height() / total))
        x = patch.get_x() + patch.get_width() / 6
        y = patch.get_y() + patch.get_height() + 0.5
        plot.annotate(percentage, (x, y), size=12)
    title = col_name + " Distribution"
    plt.title(title)
    plt.xlabel(col_name)
    plt.savefig(title + ".png")
    plt.show()

def plot_relation_to_gender(data, col_name):
    plot = sns.catplot(x=col_name, hue='gender', data=data, kind="count")
    plot.fig.set_size_inches(20, 8)
    title = "The relationship between gender to " + col_name
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()

def get_values_name_by_prob_percentage_population(data_col, data_with_probs, prob_treshold, treshold_percentage):
    list_values_prob = list(data_with_probs[data_with_probs >= prob_treshold].dropna().index)
    data_value_counts = data_col.value_counts() / data_col.shape[0]
    current_values_small_sample = set(data_value_counts[data_value_counts <= treshold_percentage].index)
    return list(current_values_small_sample.intersection(list_values_prob))

def get_values_by_gender_prob(data_col, target):
    encoder = TargetEncoder()
    encoded_col = encoder.fit_transform(data_col, target)
    encoded_col.index = data_col
    encoded_col_groupby_prob_female = encoded_col.groupby(level=0).mean()
    encoded_col_groupby_prob_male = 1 - encoded_col_groupby_prob_female
    return encoded_col_groupby_prob_female, encoded_col_groupby_prob_male

def merge_values_by_dict(data, value_dict, name_new_value):
    for key in value_dict.keys():
        if len(value_dict[key]) <= 1:
            continue
        data[key].mask(data[key].isin(value_dict[key]), name_new_value, inplace=True)
    return data

def get_percentage_missing_values_and_mean_prob_tag(data):
    data.rename(columns=lambda x: x.split('_')[0], inplace=True)
    mean_prob_tag = data.mean(skipna=True).reset_index(drop=True)
    numeric_cols_percentage_missing_values = (data.isnull().sum() / len(data)).reset_index(drop=True)
    tags_name = pd.Series(data.columns)
    new_data = pd.concat([tags_name, mean_prob_tag, numeric_cols_percentage_missing_values], axis=1, keys=["Tag", "The average probability", "Percentage of missing values"])
    return new_data


# IMPORT DATA
json_data = pd.read_json("jds_data_16102020.json", convert_dates=True)
data = pd.concat([json_data.iloc[:, 0:2], pd.json_normalize(json_data.iloc[:, 2])], axis=1)  # convert the third col (dict of features) to 26 cols of features

data["title"].nunique() # new!

# DATA UNDERSTANDING
print(data.info())
## remove chaet col: "Title"
data = data.iloc[:, 1:]

gender_colors = {'Female': 'blue', 'Male': 'orange'} # new!

## remove rows without any information
data.dropna(how="all", subset=data.columns[2:], inplace=True)  # left 39329 rows (of 50K)
num_nan_each_row = data.isnull().sum(axis=1).sort_values()

## hist of numeric variables
plt.hist(num_nan_each_row)
plt.xlabel('NANs in row')
plt.ylabel('Count of rows')
plt.title('Histogram of NANs in row')
plt.xticks(range(0, 25))
plt.grid(True)
plt.savefig("Histogram of NANs in row.png")

## cols - missing values
counted_missing_values = data.isnull().sum()
cols_percentage_missing_values = data.isnull().sum() / len(data) * 100

## target
data["gender"].value_counts()  # Female 33328 (~85%); Male 6001 (~15%)

# bar plot - gender distribution new!
plot = sns.countplot(x=data["gender"])
total = len(data["gender"])
for patch in plot.patches:
    percentage = '{:.1f}%'.format(100 * (patch.get_height() / total))
    x = patch.get_x() + patch.get_width() / 3
    y = patch.get_y() + patch.get_height() + 10
    plot.annotate(percentage, (x, y), size=12)
title = "Gender Distribution"
plt.title(title)
plt.xlabel("Gender")
plt.savefig(title + ".png")

## numeric cols
numeric_variables_describe = data.describe().transpose()
data.gender = pd.Series(np.where(data.gender.values == 'Female', 1, 0), data.index)

# VISUALIZATION FOR SALES TEAM
male_data = data[data["gender"] == 0]
female_data = data[data["gender"] == 1]
numeric_variables_names = numeric_variables_describe.index # new!
female_percentage_missing_values_mean_prob_tag = get_percentage_missing_values_and_mean_prob_tag(data=female_data.loc[:, numeric_variables_names])
male_percentage_missing_values_mean_prob_tag = get_percentage_missing_values_and_mean_prob_tag(data=male_data.loc[:, numeric_variables_names])

missing_value_tags = pd.concat([female_percentage_missing_values_mean_prob_tag["Percentage of missing values"], male_percentage_missing_values_mean_prob_tag["Percentage of missing values"]], axis=1, keys=["Female", "Male"])
missing_value_tags.index = female_percentage_missing_values_mean_prob_tag["Tag"]
missing_value_tags.plot.bar()
# Set descriptions:
plt.title('The percentage of missing values in each tag')
plt.ylabel('Percentage of missing values')
plt.xlabel('Tags')
plt.grid(color='lightgray')
plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig("The percentage of missing values in each tag.png")

prob_data = pd.concat([female_percentage_missing_values_mean_prob_tag["The average probability"], male_percentage_missing_values_mean_prob_tag["The average probability"]], axis=1, keys=["Female", "Male"])
prob_data.index = female_percentage_missing_values_mean_prob_tag["Tag"]
prob_data.plot.bar()
# Set descriptions:
plt.title('The average probability of each tag')
plt.ylabel('The average probability')
plt.xlabel('Tags')
plt.grid(color='lightgray')
plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig("The average probability of each tag")

## categorical cols
categorical_col_names = list(data.select_dtypes(['object']).columns)
data[categorical_col_names].nunique()
data[categorical_col_names].nunique().sum() # 418

data["Cat"].value_counts()  # new!

## fill nan values
data.loc[:, categorical_col_names] = data.loc[:, categorical_col_names].fillna("notFound")
data.loc[:, numeric_variables_describe.index] = data.loc[:, numeric_variables_describe.index].fillna(1)

# merge values to specific gender
treshold_percentage = 0.03
threshold_prob_male = 0.20
threshold_prob_female = 0.95
dict_values_merge_male_by_prob = dict()
dict_values_merge_female_by_prob = dict()
dict_values_merge_definitely_male = dict()
dict_values_merge_definitely_female = dict()
for col in categorical_col_names:
    encoded_col_groupby_prob_female, encoded_col_groupby_prob_male = get_values_by_gender_prob(data[col], data["gender"])
    dict_values_merge_female_by_prob[col] = get_values_name_by_prob_percentage_population(data_col=data[col], data_with_probs=encoded_col_groupby_prob_female, prob_treshold=threshold_prob_female, treshold_percentage=treshold_percentage)
    dict_values_merge_male_by_prob[col] = get_values_name_by_prob_percentage_population(data_col=data[col], data_with_probs=encoded_col_groupby_prob_male, prob_treshold=threshold_prob_male, treshold_percentage=treshold_percentage)
    dict_values_merge_definitely_female[col] = get_values_name_by_prob_percentage_population(data_col=data[col], data_with_probs=encoded_col_groupby_prob_female, prob_treshold=0.98, treshold_percentage=100)
    dict_values_merge_definitely_male[col] = get_values_name_by_prob_percentage_population(data_col=data[col], data_with_probs=encoded_col_groupby_prob_male, prob_treshold=0.98, treshold_percentage=100)

data = merge_values_by_dict(data, dict_values_merge_definitely_male, "definitely_male")
data = merge_values_by_dict(data, dict_values_merge_male_by_prob, "low_sample_male")
data = merge_values_by_dict(data, dict_values_merge_definitely_female, "definitely_female")
data = merge_values_by_dict(data=data, value_dict=dict_values_merge_female_by_prob, name_new_value="low_sample_female")

# get list of value with low sample
treshold_size_low_sample = 10
values_to_remove = []
for col in categorical_col_names:
    data_value_counts = data[col].value_counts()
    list_values =list(data_value_counts[data_value_counts <= treshold_size_low_sample].index)
    for element in list_values:
        values_to_remove.append(col + "_" + element)

# VISUALIZATION
for col in categorical_col_names:
    plot_categorical_distribution(data[col], col)
    plot_relation_to_gender(data, col)

# ONE HOT ENCODING
data_dummies = pd.concat([data.iloc[:, 0:1], pd.get_dummies(data[categorical_col_names])], axis=1)

## removing values with low sample size
cols_to_remove = []
for col in data_dummies.columns[1:]:
    if col in values_to_remove:
        cols_to_remove.append(col)
data_dummies = data_dummies.drop(cols_to_remove, axis=1)

# SPLIT THE DATA
X = data_dummies.iloc[:, 2:].reset_index(drop=True)
y = data_dummies["gender"].reset_index(drop=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
for tr_idx, te_idx in split.split(X, y):
    X_train = X.iloc[tr_idx, :]
    X_test = X.iloc[te_idx, :]
    y_train = y.iloc[tr_idx]
    y_test = y.iloc[te_idx]

# np.round((y_train.value_counts() / y_train.shape[0]) * 100, 1)
# np.round((y_test.value_counts() / y_test.shape[0]) * 100, 1)

# MODELING
# base model
RF = RandomForestClassifier(random_state=42)

# tuned model
rf_param = {'max_depth': list(range(50, 500, 50)) + [None],
            'n_estimators': list(range(50, 400, 50)),
            'max_features': ['sqrt', 'auto', 'log2'],
            'min_samples_split': list(range(4, 200, 20)),
            'min_samples_leaf': list(range(4, 200, 20)),
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'],
            "class_weight": ['balanced', "balanced_subsample"]
            }

rf_random = RandomizedSearchCV(estimator=RF, param_distributions=rf_param, n_iter=200, cv=10, random_state=42)
rf_random.fit(X_train, y_train)

# chosen_model = RandomForestClassifier(class_weight="balanced", criterion="entropy", max_depth=450, max_features='auto', min_samples_split=4, min_samples_leaf=4, n_estimators=350, random_state=42)

chosen_model = rf_random.best_estimator_
print(chosen_model.get_params())
chosen_model.fit(X_train, y_train)
y_pred_train = chosen_model.predict(X_train)
y_pred_test = chosen_model.predict(X_test)
print("** Train evaluation **")
print("The TRAIN f1 score is " + str(np.round(f1_score(y_train, y_pred_train), 2)))
print("The TRAIN Recall score is " + str(np.round(recall_score(y_train, y_pred_train), 2)))
print("The TRAIN precision score is " + str(np.round(precision_score(y_train, y_pred_train), 2)))
print("** Test evaluation **")
evaluation(chosen_model, X_test, y_test, y_pred_test)

# plot feature importance
plot_top_feature_importance(chosen_model, X_train, top=20)

# metrics for each class
print(classification_report(y_test, y_pred_test))