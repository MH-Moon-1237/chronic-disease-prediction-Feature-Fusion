# ================================================================
# main.py
# Proposed Framework for Chronic Disease Prediction
# ================================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler


# 1. Data Collection
df = pd.read_csv("chronic_disease_prediction_dataset.csv")


# 2. Data Preprocessing
df.drop(columns=["Patient_ID"], inplace=True)

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

if len(num_cols) > 0:
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

if len(cat_cols) > 0:
    df[cat_cols] = SimpleImputer(
        strategy="most_frequent"
    ).fit_transform(df[cat_cols])


# 2.1 IQR-based outlier capping
num_only = [
    col for col in df.select_dtypes(include="number").columns
    if col != "HasChronicDisease"
]

Q1 = df[num_only].quantile(0.25)
Q3 = df[num_only].quantile(0.75)
IQR = Q3 - Q1

for col in num_only:
    lower_bound = Q1[col] - 1.5 * IQR[col]
    upper_bound = Q3[col] + 1.5 * IQR[col]
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)


# 2.2 Label Encoding
for col in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])


# 3. Feature and Target Separation
X = df.drop(columns=["HasChronicDisease"])
y = df["HasChronicDisease"]


# 4. Hybrid Feature Selection
anova_selector = SelectKBest(score_func=f_classif, k=7)
mi_selector = SelectKBest(score_func=mutual_info_classif, k=7)

anova_selector.fit(X, y)
mi_selector.fit(X, y)

anova_features = set(X.columns[anova_selector.get_support()])
mi_features = set(X.columns[mi_selector.get_support()])

selected_features = list(anova_features.union(mi_features))
X = X[selected_features]


# 5. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 6. Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 7. Random Oversampling (training partition only)
ros = RandomOverSampler(random_state=42)

X_train_balanced, y_train_balanced = ros.fit_resample(
    X_train_scaled,
    y_train
)


# 8. Polynomial Feature Generation
poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train_balanced)
X_test_poly = poly.transform(X_test_scaled)


# 9. Feature Fusion
X_train_fused = np.hstack([
    X_train_balanced,
    X_train_poly
])

X_test_fused = np.hstack([
    X_test_scaled,
    X_test_poly
])


# 10. Base Classifiers
decision_tree = DecisionTreeClassifier(
    max_depth=14,
    criterion="gini",
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    random_state=42
)

xgboost = XGBClassifier(
    n_estimators=400,
    learning_rate=0.08,
    max_depth=3,
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)


# 11. Proposed Soft-Voting Ensemble
proposed_model = VotingClassifier(
    estimators=[
        (
            "xgb",
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.08,
                max_depth=3,
                eval_metric="logloss",
                random_state=42,
                verbosity=0
            )
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                random_state=42
            )
        )
    ],
    voting="soft",
    n_jobs=-1
)


# 12. Model Training
decision_tree.fit(X_train_fused, y_train_balanced)
random_forest.fit(X_train_fused, y_train_balanced)
xgboost.fit(X_train_fused, y_train_balanced)
proposed_model.fit(X_train_fused, y_train_balanced)


# 13. Generate Predictions
y_pred_dt = decision_tree.predict(X_test_fused)
y_pred_rf = random_forest.predict(X_test_fused)
y_pred_xgb = xgboost.predict(X_test_fused)
y_pred_proposed = proposed_model.predict(X_test_fused)


# 14. Store models and data for evaluation / XAI
models = {
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "XGBoost": xgboost,
    "Proposed Model": proposed_model
}

predictions = {
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "Proposed Model": y_pred_proposed
}

test_data = {
    "X_test": X_test,
    "X_test_scaled": X_test_scaled,
    "X_test_fused": X_test_fused,
    "y_test": y_test
}

feature_information = {
    "selected_features": selected_features,
    "anova_features": list(anova_features),
    "mi_features": list(mi_features)
}
