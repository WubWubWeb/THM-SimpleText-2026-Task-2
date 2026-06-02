import torch
import pandas as pd
import json
import os
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


def make_compute_metrics(id2label):
    labels_list = [id2label[i] for i in sorted(id2label)]

    def compute_metrics(eval_pred):
        logits, true_labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        all_labels = list(id2label.keys())
        macro_f1 = f1_score(true_labels, preds, average="macro", labels=all_labels, zero_division=0)
        per_class = f1_score(
            true_labels, preds, average=None,
            labels=all_labels, zero_division=0
        )
        metrics = {"eval_macro_f1": macro_f1}
        for i, lbl in enumerate(labels_list):
            metrics[f"f1_{lbl}"] = per_class[i]
        return metrics

    return compute_metrics



class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if torch.isnan(logits).any():
            print("NaNs in logits!")
        weights = self.class_weights.to(
            device=logits.device,
            dtype=torch.float32
        )
        loss_fn = CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits.float(), labels)
        if torch.isnan(loss):
            print("NaN loss!")
        return (loss, outputs) if return_outputs else loss


class BERTCouncil:
    def __init__(self, distilbert=False, deberta=False, roberta=False, bert=False, scibert=False):
        self.model_configs = {}
        if distilbert:
            self.model_configs["distilbert"] = "distilbert-base-uncased"
        if deberta:
            self.model_configs["deberta"] = "microsoft/deberta-v3-base"
        if roberta:
            self.model_configs["roberta"] = "roberta-base"
        if bert:
            self.model_configs["bert"] = "bert-base-uncased"
        if scibert:
            self.model_configs["scibert"] = "allenai/scibert_scivocab_uncased"

        self.labels = [
            "None",
            "GENERATION_FAILURE",
            "LEAKED_INSTRUCTIONS",
            "UNGROUNDED_INJECTION",
            "REPETITIVE_CONTENT",
            "GROUNDED_OVERGENERATION",
        ]
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for i, lbl in enumerate(self.labels)}


    def flatten_data(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for entry in data:
            source_text = entry["source"]
            doc_id = entry["id"]
            for i, (sent, lbl) in enumerate(zip(entry["sentences"], entry["labels"])):
                rows.append({
                    "doc_id": doc_id,
                    "snt_id": f"{doc_id}_{i}",
                    "source_sentence": source_text,
                    "simplified_sentence": sent,
                    "label_string": lbl,
                    "label": self.label2id.get(lbl, 0),
                })
        return pd.DataFrame(rows)

    def compute_class_weights(self, df):
        counts = np.bincount(df["label"].values, minlength=len(self.labels)).astype(float)
        counts = np.where(counts == 0, 1, counts)          # avoid div-by-zero
        weights = 1.0 / np.log(counts + 1) # alternativ 1.0 / counts  ODER  np.sqrt(1.0 / counts)
        weights = weights / weights.sum() * len(self.labels)
        return torch.tensor(weights, dtype=torch.float)

    def prepare_dataset(self, df, tokenizer, is_train=True):
        data = {
            "source": df["source_sentence"].tolist(),
            "simplified": df["simplified_sentence"].tolist(),
        }
        if is_train:
            data["labels"] = df["label"].tolist()

        ds = Dataset.from_dict(data)
        ds = ds.map(
            lambda e: tokenizer(
                text=e["source"],
                text_pair=e["simplified"],
                truncation=True,
                max_length=256
            ),
            batched=True,
        )

        columns = ["input_ids", "attention_mask"]

        if "token_type_ids" in ds.column_names:
            columns.append("token_type_ids")#ds = ds.remove_columns(["token_type_ids"])
        
        if is_train:
            columns.append("labels")

        ds.set_format(type="torch", columns=columns)
        return ds


    def aggregate_to_doc_level(self, df_preds):
        doc_results = []
        non_none_labels = [l for l in self.labels if l != "None"]

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
                    lbl: group[f"{lbl}_prob"].sum() for lbl in non_none_labels
                }
                best_label = max(class_sums, key=class_sums.get)
                max_prob = group[
                    [f"{l}_prob" for l in non_none_labels]
                ].values.max()

                doc_results.append({
                    "doc_id": doc_id,
                    "overgeneration_detected": True,
                    "doc_label": best_label,
                    "max_error_prob": float(max_prob),
                })

        return doc_results

    def run_all_models(
        self,
        train_path,
        test_path,
        team_name="TeamTEAM",
        sample_size=None,
        val_split=0.1,
        train_size=2,
        eval_size=2
    ):
        train_df_full = self.flatten_data(train_path)
        test_df = self.flatten_data(test_path)

        if sample_size:
            train_df_full = train_df_full.sample(
                n=min(sample_size, len(train_df_full)), random_state=42
            )
        label_counts = train_df_full["label"].value_counts()

        if label_counts.min() >= 2:
            stratify_labels = train_df_full["label"]
        else:
            stratify_labels = None
            print("Warning: disabling stratification due to rare classes")

        train_df, val_df = train_test_split(
            train_df_full,
            test_size=val_split,
            random_state=42,
            stratify=stratify_labels,
        )


        class_weights = self.compute_class_weights(train_df)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        compute_metrics = make_compute_metrics(self.id2label)

        for model_key, model_path in self.model_configs.items():
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                pad_to_multiple_of=8
            )

            train_ds = self.prepare_dataset(train_df, tokenizer, is_train=True)
            val_ds = self.prepare_dataset(val_df, tokenizer, is_train=True)
            test_ds = self.prepare_dataset(test_df, tokenizer, is_train=False)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id,
            )
            model.to(device)

            n_epochs = 4 if model_key == "scibert" else 3

            checkpoint_dir = f"/content/drive/MyDrive/colab_data/results_{model_key}"

            args = TrainingArguments(
                output_dir=checkpoint_dir,
                per_device_train_batch_size=train_size,
                per_device_eval_batch_size=eval_size,
                gradient_accumulation_steps=2,
                num_train_epochs=n_epochs,

                learning_rate=2e-5 if model_key == "roberta" else 1e-5,
                weight_decay=0.01,
                warmup_ratio=0.1,
                max_grad_norm=1.0,

                eval_strategy="steps",
                save_strategy="steps",
                logging_strategy="steps",
                eval_steps=4500,
                save_steps=4500,
                logging_steps=4500,
                
                save_total_limit=2,
                load_best_model_at_end=True,

                metric_for_best_model="eval_macro_f1",
                greater_is_better=True,

                fp16=torch.cuda.is_available(),
            )

            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
            )
            if os.path.exists(checkpoint_dir) and any(os.scandir(checkpoint_dir)):
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()

            predictions = trainer.predict(test_ds)
            logits = torch.tensor(predictions.predictions, dtype=torch.float32)
            probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

            submission_records = []
            for i, (_, row) in enumerate(test_df.iterrows()):
                record = {
                    "doc_id": row["doc_id"],
                    "snt_id": row["snt_id"],
                    "run_id": f"{team_name}_{model_key}",
                }
                for idx, label in enumerate(self.labels):
                    record[f"{label}_prob"] = float(probs[i][idx])
                record["predicted_label"] = self.labels[int(np.argmax(probs[i]))]
                submission_records.append(record)

            df_preds = pd.DataFrame(submission_records)

            val_predictions = trainer.predict(val_ds)
            val_preds = np.argmax(val_predictions.predictions, axis=-1)
            val_true = val_df["label"].values
            print(classification_report(
                val_true, val_preds,
                labels=list(range(len(self.labels))),
                target_names=self.labels,
                zero_division=0
            ))
            doc_results = self.aggregate_to_doc_level(df_preds)

            with open(f"submission_{model_key}_sentences.json", "w") as f:
                json.dump(submission_records, f, indent=4)

            with open(f"submission_{model_key}_docs.json", "w") as f:
                json.dump(doc_results, f, indent=4)
            del model
            del trainer
            torch.cuda.empty_cache()
        return "All models complete."