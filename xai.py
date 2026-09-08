# SHAP and LIME explainability

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import lime
import lime.lime_tabular

from sklearn.preprocessing import PolynomialFeatures

from main import models, test_data, feature_information


os.makedirs("results/figures", exist_ok=True)


# Proposed model
model = models["Proposed Model"]
xgb_model = model.estimators_[0]

X_test_fused = test_data["X_test_fused"]
X_test_scaled = test_data["X_test_scaled"]
X_train_scaled = test_data["X_train_scaled"]

selected_features = feature_information["selected_features"]


# Feature names
poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

poly.fit(np.zeros((1, len(selected_features))))

poly_features = list(poly.get_feature_names_out(selected_features))
feature_names = selected_features + poly_features


# SHAP summary plots
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_fused)

plt.figure()
shap.summary_plot(
    shap_values,
    X_test_fused,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(
    "results/figures/shap_class1_has_disease.png",
    dpi=1000,
    bbox_inches="tight"
)
plt.close()

plt.figure()
shap.summary_plot(
    -shap_values,
    X_test_fused,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(
    "results/figures/shap_class0_no_disease.png",
    dpi=1000,
    bbox_inches="tight"
)
plt.close()


# SHAP force plots
background = shap.sample(
    X_test_fused,
    min(100, len(X_test_fused)),
    random_state=42
)

force_explainer = shap.TreeExplainer(
    xgb_model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability"
)

for sample_index, output_name in [
    (4, "shap_force_class0"),
    (34, "shap_force_class1")
]:
    if sample_index >= len(X_test_fused):
        continue

    shap_exp = force_explainer(
        X_test_fused[sample_index:sample_index + 1]
    )

    html_path = f"results/figures/{output_name}.html"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(
            shap.plots.force(
                shap_exp[0],
                matplotlib=False
            ).html()
        )


# LIME
def lime_predict_proba(X):
    X = np.asarray(X)
    X_poly = poly.transform(X)
    X_fused = np.hstack([X, X_poly])
    return model.predict_proba(X_fused)


lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=selected_features,
    class_names=["No", "Yes"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

class_info = {
    0: {"name": "No"},
    1: {"name": "Yes"}
}

all_class_data = {}

for cls, info in class_info.items():
    feature_weights = defaultdict(list)

    for i in range(len(X_test_scaled)):
        explanation = lime_explainer.explain_instance(
            X_test_scaled[i],
            lime_predict_proba,
            num_features=10,
            labels=[cls]
        )

        for feat, weight in explanation.as_list(label=cls):
            feature_weights[feat].append(weight)

    top_features = sorted(
        {
            feat: np.mean(weights)
            for feat, weights in feature_weights.items()
        }.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    all_class_data[cls] = {
        "features": [x[0] for x in top_features][::-1],
        "weights": [x[1] for x in top_features][::-1],
        "info": info
    }

    print(f"Class {cls} ({info['name']}) done!")


fig, axes = plt.subplots(1, 2, figsize=(18, 9))

for ax, (cls, data), title in zip(
    axes,
    all_class_data.items(),
    ["(a) Class: No", "(b) Class: Yes"]
):
    weights = data["weights"]
    features = data["features"]

    colors = [
        "#27ae60" if w >= 0 else "#e74c3c"
        for w in weights
    ]

    bars = ax.barh(
        features,
        weights,
        color=colors,
        height=0.65,
        edgecolor="white",
        linewidth=0.5
    )

    for bar, weight in zip(bars, weights):
        xpos = weight + (0.0008 if weight >= 0 else -0.0008)

        ax.text(
            xpos,
            bar.get_y() + bar.get_height() / 2,
            f"{weight:.4f}",
            va="center",
            ha="left" if weight >= 0 else "right",
            fontsize=8.5,
            fontweight="bold"
        )

    ax.axvline(0, linewidth=1.5)

    pos_patch = mpatches.Patch(
        facecolor="#27ae60",
        edgecolor="white",
        label=f"Supports {data['info']['name']}"
    )

    neg_patch = mpatches.Patch(
        facecolor="#e74c3c",
        edgecolor="white",
        label=f"Suppresses {data['info']['name']}"
    )

    ax.legend(
        handles=[pos_patch, neg_patch],
        fontsize=9,
        loc="lower right",
        framealpha=0.9
    )

    ax.set_xlabel(
        "Mean LIME Feature Contribution Weight",
        fontsize=11
    )
    ax.set_ylabel("Feature Condition", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


plt.tight_layout(w_pad=4)

plt.savefig(
    "results/figures/Fig_LIME_HybridFusion.png",
    dpi=1200,
    bbox_inches="tight"
)
plt.close()

print("SHAP and LIME analysis completed.")
print("Saved files in results/figures/")
