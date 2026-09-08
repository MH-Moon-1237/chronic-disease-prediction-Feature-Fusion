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
from sklearn.utils import resample
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

num_only = [c for c in df.select_dtypes(include="number").columns
            if c != "HasChronicDisease"]

Q1 = df[num_only].quantile(0.25)
Q3 = df[num_only].quantile(0.75)
IQR = Q3 - Q1

for col in num_only:
    df[col] = df[col].clip(
        lower=Q1[col] - 1.5 * IQR[col],
        upper=Q3[col] + 1.5 * IQR[col]
    )

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

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


# Helper functions
def apply_ros(Xtr, ytr):
    d = Xtr.copy()
    d["t"] = ytr.values

    maj = d[d.t == 0]
    mn = d[d.t == 1]

    mn2 = resample(
        mn,
        replace=True,
        n_samples=len(maj),
        random_state=42
    )

    db = pd.concat([maj, mn2])

    return db.drop(columns=["t"]), db["t"]


def get_fused(Xtr, Xte):
    sc = StandardScaler()

    A = sc.fit_transform(Xtr)
    B = sc.transform(Xte)

    poly = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )

    Ap = poly.fit_transform(A)
    Bp = poly.transform(B)

    sc2 = StandardScaler()

    Ap = sc2.fit_transform(Ap)
    Bp = sc2.transform(Bp)

    return np.hstack([A, Ap]), np.hstack([B, Bp])


def get_raw_scaled(Xtr, Xte):
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte)


def get_models(n_est=200, max_d=10, lr=0.08, ratio=1.0):
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_est,
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
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=3,
                    scale_pos_weight=ratio,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0
                )),
                ("rf", RandomForestClassifier(
                    n_estimators=n_est,
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
                n_est=200, max_d=10, lr=0.08, ratio=1.0):

    print("\n" + "=" * 77)
    print(title)
    print("=" * 77)

    for name, model in get_models(
        n_est, max_d, lr, ratio
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

Xtr_f, Xte_f = get_fused(Xtr, Xte)

print_table(
    "4.3.1 Performance on Imbalanced Data (Without ROS)",
    Xtr_f, Xte_f, ytr, yte,
    n_est=200, max_d=10, lr=0.08, ratio=ratio
)


# 4.3.2 All features
X = X_all.copy()

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr2, ytr2 = apply_ros(Xtr, ytr)
Xtr_f, Xte_f = get_fused(Xtr2, Xte)

print_table(
    "4.3.2 Performance on All Features (Without Feature Selection)",
    Xtr_f, Xte_f, ytr2, yte,
    n_est=200, max_d=10, lr=0.08
)


# 4.3.3 ANOVA-only selection
X = X_all[sel_anova]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr2, ytr2 = apply_ros(Xtr, ytr)
Xtr_f, Xte_f = get_fused(Xtr2, Xte)

print_table(
    "4.3.3 Performance on ANOVA-Only Feature Selection",
    Xtr_f, Xte_f, ytr2, yte,
    n_est=200, max_d=10, lr=0.08
)


# 4.3.4 ROS
X = X_all[sel_both]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr2, ytr2 = apply_ros(Xtr, ytr)
Xtr_f, Xte_f = get_fused(Xtr2, Xte)

print_table(
    "4.3.4 Performance on Class Balancing With Random OverSampling",
    Xtr_f, Xte_f, ytr2, yte,
    n_est=200, max_d=10, lr=0.08
)


# 4.3.5 Raw features
X = X_all[sel_both]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr2, ytr2 = apply_ros(Xtr, ytr)
Xtr_f, Xte_f = get_raw_scaled(Xtr2, Xte)

print_table(
    "4.3.5 Performance on Raw Features Without Fusion",
    Xtr_f, Xte_f, ytr2, yte,
    n_est=200, max_d=10, lr=0.08
)


# 4.3.6 Full pipeline
X = X_all[sel_both]

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtr2, ytr2 = apply_ros(Xtr, ytr)
Xtr_f, Xte_f = get_fused(Xtr2, Xte)

print_table(
    "4.3.6 Performance on Hybrid Feature Fusion With Heterogeneous Ensemble Learning",
    Xtr_f, Xte_f, ytr2, yte,
    n_est=500, max_d=10, lr=0.08
)
