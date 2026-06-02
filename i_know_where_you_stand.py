import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

LABELS = [
    "None",
    "GENERATION_FAILURE",
    "LEAKED_INSTRUCTIONS",
    "UNGROUNDED_INJECTION",
    "REPETITIVE_CONTENT",
    "GROUNDED_OVERGENERATION",
]

NON_NONE_LABELS = [l for l in LABELS if l != "None"]


def load_ground_truth(train_json_path):
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)

    rows = []
    for entry in train_raw:
        doc_id = entry["id"]
        source_text = entry["source"]
        src_word_count = len(source_text.split())

        for i, (sent, lbl) in enumerate(zip(entry["sentences"], entry["labels"])):
            simp_word_count = len(sent.split())
            row = {
                "snt_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source_sentence": source_text,
                "simplified_sentence": sent,
                "abs_len": simp_word_count,
                "len_ratio": simp_word_count / src_word_count if src_word_count > 0 else 1.0,
                "is_tiny": 1 if simp_word_count < 6 else 0,
                "true_label": lbl,
                "true_label_idx": LABELS.index(lbl),
            }
            for l in LABELS:
                row[l] = 1 if lbl == l else 0
            rows.append(row)

    return pd.DataFrame(rows)


def merge_model_predictions(df_gt, model_files):
    df_merged = df_gt.copy()

    for file_path in model_files:
        model_name = (
            os.path.basename(file_path)
            .replace("submission_", "")
            .split(".")[0]
            .replace("_sentences_DEV", "")
            .replace("_sentences", "")
        )
        with open(file_path, "r") as f:
            preds = json.load(f)

        df_preds = pd.DataFrame(preds)

        prob_cols = [
            col for col in df_preds.columns
            if col.endswith("_prob")
        ]

        df_preds[f"model_present_{model_name}"] = (
            df_preds[prob_cols]
            .sum(axis=1)
            .gt(0)
            .astype(int)
        )
        df_preds = df_preds[
            ["snt_id"] + prob_cols + [f"model_present_{model_name}"]
            ].rename(
            columns={c: f"{c}_{model_name}" for c in prob_cols}
        )

        df_merged = pd.merge(
            df_merged,
            df_preds,
            on="snt_id",
            how="left"
        )

        df_merged = df_merged.fillna(0.0)

    if len(df_merged) == 0:
        raise ValueError(
            "Merge produced 0 rows — check that snt_id formats match between "
            "the ground-truth JSON and the prediction files."
        )
    return df_merged

def build_feature_matrix(df):
    prob_features = [
        c for c in df.columns
        if "_prob_" in c
        and "distilbert" not in c
    ]
    presence_features = [
        c for c in df.columns
        if c.startswith("model_present_")
    ]
    feature_cols = prob_features + presence_features

    return df[feature_cols].values, feature_cols

def create_meta_ensemble_oof(model_files, train_json_path, n_splits=5):
    df_gt = load_ground_truth(train_json_path)
    df_merged = merge_model_predictions(df_gt, model_files)

    X, feature_cols = build_feature_matrix(df_merged)
    y = df_merged["true_label_idx"].values


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    oof_preds = cross_val_predict(meta_model, X, y, cv=skf, method="predict")

    print(classification_report(y, oof_preds, target_names=LABELS, zero_division=0))

    meta_model.fit(X, y)

    plot_model_importance(meta_model, feature_cols)
    return meta_model, df_merged, feature_cols


def create_meta_ensemble_dev(model_files, dev_json_path):
    df_gt = load_ground_truth(dev_json_path)
    df_merged = merge_model_predictions(df_gt, model_files)

    X, feature_cols = build_feature_matrix(df_merged)
    y = df_merged["true_label_idx"].values

    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    meta_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    meta_model.fit(X, y)

    preds = meta_model.predict(X)
    print("\n=== Dev Classification Report ===")
    print(classification_report(y, preds, target_names=LABELS, zero_division=0))

    plot_model_importance(meta_model, feature_cols)
    return meta_model, df_merged, feature_cols


def run_inference_and_save(
    meta_model,
    test_model_files,
    test_json_path,
    feature_cols,
    output_file="submission_meta_ensemble.json",
    team_name="TeamTEAM",
):
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    test_rows = []
    for entry in test_raw:
        doc_id = entry["id"]
        source_text = entry["source"]
        src_word_count = len(source_text.split())
        for i, sent in enumerate(entry["sentences"]):
            simp_word_count = len(sent.split())
            test_rows.append({
                "snt_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source_sentence": source_text,
                "simplified_sentence": sent,
                "abs_len": simp_word_count,
                "len_ratio": simp_word_count / src_word_count if src_word_count > 0 else 1.0,
                "is_tiny": 1 if simp_word_count < 6 else 0,
            })
    df_test = pd.DataFrame(test_rows)

    for file_path in test_model_files:
        model_name = (
            os.path.basename(file_path)
            .replace("submission_", "")
            .split(".")[0]
            .replace("_sentences", "")
        )

        with open(file_path, "r") as f:
            preds = json.load(f)

        df_preds = pd.DataFrame(preds)

        prob_cols = [col for col in df_preds.columns if col.endswith("_prob")]

        # --- ADD THIS (critical fix) ---
        df_preds[f"model_present_{model_name}"] = (
            df_preds[prob_cols].sum(axis=1).gt(0).astype(int)
        )

        rename_map = {c: f"{c}_{model_name}" for c in prob_cols}
        rename_map[f"model_present_{model_name}"] = f"model_present_{model_name}"

        df_preds = df_preds[["snt_id"] + prob_cols + [f"model_present_{model_name}"]].rename(columns=rename_map)

        df_test = pd.merge(df_test, df_preds, on="snt_id", how="left")
        df_test = df_test.fillna(0.0)

    X_test = df_test[feature_cols].values

    if np.isnan(X_test).any():
        nan_cols = [feature_cols[i] for i in range(X_test.shape[1]) if np.isnan(X_test[:, i]).any()]
        print(f"Warning: NaN found in columns: {nan_cols}")
        X_test = np.nan_to_num(X_test, nan=0.0)

    ensemble_probs = meta_model.predict_proba(X_test)
    final_preds = meta_model.predict(X_test)

    meta_records = []
    for i, (_, row) in enumerate(df_test.iterrows()):
        record = {
            "snt_id": row["snt_id"],
            "doc_id": row["doc_id"],
            "source_sentence": row["source_sentence"],
            "simplified_sentence": row["simplified_sentence"],
            "run_id": f"{team_name}_meta_ensemble",
            "predicted_label": LABELS[final_preds[i]],
        }
        for idx, label in enumerate(LABELS):
            record[f"{label}_prob"] = float(ensemble_probs[i][idx])

        model_names = list(dict.fromkeys(
            col.split("_prob_")[1]
            for col in df_test.columns
            if "_prob_" in col
        ))
        for model_name in model_names:
            record[f"bert_{model_name}"] = {
                label: float(row[f"{label}_prob_{model_name}"])
                for label in LABELS
                if f"{label}_prob_{model_name}" in df_test.columns
            }
        meta_records.append(record)

    df_meta = pd.DataFrame(meta_records)
    doc_results = aggregate_to_doc_level(df_meta)

    with open(output_file, "w") as f:
        json.dump(meta_records, f, indent=4)

    doc_output = output_file.replace(".json", "_docs.json")
    with open(doc_output, "w") as f:
        json.dump(doc_results, f, indent=4)

    print(f"Saved {len(meta_records)} sentence records → {output_file}")
    print(f"Saved {len(doc_results)} document records → {doc_output}")


def aggregate_to_doc_level(df_preds):
    doc_results = []
    for doc_id, group in df_preds.groupby("doc_id"):
        none_mask = group["predicted_label"] == "None"
        if none_mask.all():
            doc_results.append({
                "doc_id": doc_id,
                "overgeneration_detected": False,
                "doc_label": "None",
                "max_error_prob": 0.0,
            })
        else:
            class_sums = {
                lbl: group[f"{lbl}_prob"].sum() for lbl in NON_NONE_LABELS
            }
            best_label = max(class_sums, key=class_sums.get)
            max_prob = group[
                [f"{l}_prob" for l in NON_NONE_LABELS]
            ].values.max()
            doc_results.append({
                "doc_id": doc_id,
                "overgeneration_detected": True,
                "doc_label": best_label,
                "max_error_prob": float(max_prob),
            })
    return doc_results


def plot_model_importance(meta_model, feature_names):
    importances = meta_model.feature_importances_
    model_contribution = {}
    for name, imp in zip(feature_names, importances):
        m_name = name.split("_")[-1]
        model_contribution[m_name] = model_contribution.get(m_name, 0) + imp

    names = list(model_contribution.keys())
    values = list(model_contribution.values())

    plt.figure(figsize=(10, 6))
    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#8E44AD", "#999999"]
    plt.bar(names, values, color=colors[: len(names)])
    plt.title("Which BERT Model Does the Council Trust Most?", fontsize=14)
    plt.ylabel("Total Importance Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()