# Ablation study

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from collections import Counter
from xgboost import XGBClassifier


# Base preprocessing
df = pd.read_csv("chronic_disease_prediction_dataset.csv")
df.drop(columns=["Patient_ID"], inplace=True)

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

if len(num_cols) > 0:
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

if len(cat_cols) > 0:
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

num_only = [
    c for c in df.select_dtypes(include="number").columns
    if c != "HasChronicDisease"
]

Q1 = df[num_only].quantile(0.25)
Q3 = df[num_only].quantile(0.75)
IQR = Q3 - Q1

for col in num_only:
    df[col] = df[col].clip(
        lower=Q1[col] - 1.5 * IQR[col],
        upper=Q3[col] + 1.5 * IQR[col]
    )

for col in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

X_all = df.drop(columns=["HasChronicDisease"])
y = df["HasChronicDisease"]


# Feature selection
fa = set(
    X_all.columns[
        SelectKBest(f_classif, k=7).fit(X_all, y).get_support()
    ]
)

fm = set(
    X_all.columns[
        SelectKBest(mutual_info_classif, k=7).fit(X_all, y).get_support()
    ]
)

sel_both = [
    f for f, c in Counter(list(fa) + list(fm)).items()
    if c >= 1
]

sel_anova = list(fa)


# Pipeline helpers
def apply_ros(Xtr, ytr):
    ros = RandomOverSampler(random_state=42)
    return ros.fit_resample(Xtr, ytr)


def prepare_fused(Xtr, Xte, ytr, use_ros=False):
    # Manuscript pipeline: Standardization -> optional ROS -> Polynomial -> Fusion
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    ytr_out = ytr
    if use_ros:
        ros = RandomOverSampler(random_state=42)
        Xtr_scaled, ytr_out = ros.fit_resample(Xtr_scaled, ytr)

    poly = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )

    Xtr_poly = poly.fit_transform(Xtr_scaled)
    Xte_poly = poly.transform(Xte_scaled)

    Xtr_fused = np.hstack([Xtr_scaled, Xtr_poly])
    Xte_fused = np.hstack([Xte_scaled, Xte_poly])

    return Xtr_fused, Xte_fused, ytr_out


def get_raw_scaled(Xtr, Xte, ytr, use_ros=True):
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)

    ytr_out = ytr
    if use_ros:
        ros = RandomOverSampler(random_state=42)
        Xtr_scaled, ytr_out = ros.fit_resample(Xtr_scaled, ytr)

    return Xtr_scaled, Xte_scaled, ytr_out


def get_models(rf_n_est=200, xgb_n_est=200, max_d=10, lr=0.08, ratio=1.0):
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_n_est,
            max_depth=max_d,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgb_n_est,
            learning_rate=lr,
            max_depth=3,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        ),
        "Proposed Model": VotingClassifier(
            estimators=[
                ("xgb", XGBClassifier(
                    n_estimators=xgb_n_est,
                    learning_rate=lr,
                    max_depth=3,
                    scale_pos_weight=ratio,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0
                )),
                ("rf", RandomForestClassifier(
                    n_estimators=rf_n_est,
                    max_depth=max_d,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42
                ))
            ],
            voting="soft",
            n_jobs=-1
        )
    }


def print_table(title, Xtr_f, Xte_f, ytr, yte,
                rf_n_est=200, xgb_n_est=200, max_d=10, lr=0.08, ratio=1.0):

    print("\n" + "=" * 77)
    print(title)
    print("=" * 77)

    for name, model in get_models(
        rf_n_est, xgb_n_est, max_d, lr, ratio
    ).items():

        model.fit(Xtr_f, ytr)

        yp = model.predict(Xte_f)
        yb = model.predict_proba(Xte_f)[:, 1]

        acc = accuracy_score(yte, yp)
        prec = precision_score(yte, yp, average="weighted")
        rec = recall_score(yte, yp, average="weighted")
        f1 = f1_score(yte, yp, average="weighted")
        auc = roc_auc_score(yte, yb)

        print(
            f"{name:<25} "
            f"{acc * 100:>7.2f}% "
            f"{prec:>8.4f} "
            f"{rec:>8.4f} "
            f"{f1:>8.4f} "
            f"{auc:>8.4f}"
        )


# 4.3.1 Imbalanced data
X = X_all[sel_both]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ratio = (ytr == 0).sum() / (ytr == 1).sum()

Xtr_f, Xte_f, ytr_f = prepare_fused(
    Xtr, Xte, ytr, use_ros=False
)

print_table(
    "4.3.1 Performance on Imbalanced Data (Without ROS)",
    Xtr_f, Xte_f, ytr_f, yte,
    rf_n_est=200, xgb_n_est=200, max_d=10, lr=0.08, ratio=ratio
)


# 4.3.2 All features
X = X_all.copy()

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr_f, Xte_f, ytr_f = prepare_fused(
    Xtr, Xte, ytr, use_ros=True
)

print_table(
    "4.3.2 Performance on All Features (Without Feature Selection)",
    Xtr_f, Xte_f, ytr_f, yte,
    rf_n_est=200, xgb_n_est=200, max_d=10, lr=0.08
)


# 4.3.3 ANOVA-only selection
X = X_all[sel_anova]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr_f, Xte_f, ytr_f = prepare_fused(
    Xtr, Xte, ytr, use_ros=True
)

print_table(
    "4.3.3 Performance on ANOVA-Only Feature Selection",
    Xtr_f, Xte_f, ytr_f, yte,
    rf_n_est=200, xgb_n_est=200, max_d=10, lr=0.08
)


# 4.3.4 ROS
X = X_all[sel_both]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr_f, Xte_f, ytr_f = prepare_fused(
    Xtr, Xte, ytr, use_ros=True
)

print_table(
    "4.3.4 Performance on Class Balancing With Random OverSampling",
    Xtr_f, Xte_f, ytr_f, yte,
    rf_n_est=200, xgb_n_est=200, max_d=10, lr=0.08
)


# 4.3.5 Raw features
X = X_all[sel_both]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr_f, Xte_f, ytr_f = get_raw_scaled(
    Xtr, Xte, ytr, use_ros=True
)

print_table(
    "4.3.5 Performance on Raw Features Without Fusion",
    Xtr_f, Xte_f, ytr_f, yte,
    rf_n_est=200, xgb_n_est=200, max_d=10, lr=0.08
)


# 4.3.6 Full pipeline
X = X_all[sel_both]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr_f, Xte_f, ytr_f = prepare_fused(
    Xtr, Xte, ytr, use_ros=True
)

print_table(
    "4.3.6 Performance on Feature Fusion With Ensemble Learning",
    Xtr_f, Xte_f, ytr_f, yte,
    rf_n_est=500, xgb_n_est=400, max_d=10, lr=0.08
)
