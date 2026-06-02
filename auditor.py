import pandas as pd
import json

PLACEHOLDER = "empty, due to none"

class AlignmentAuditor:
    BERT_FLAG_THRESHOLD = 0.3

    def __init__(self, merged_json_path):
        with open(merged_json_path, "r") as f:
            self.data = pd.DataFrame(json.load(f))

        self.taxonomy_keywords = {
            "GENERATION_FAILURE": [
                "loop", "repeat", "broken", "incomplete", "failed",
                "degenerate", "corrupted",
            ],
            "LEAKED_INSTRUCTIONS": [
                "assistant", "prompt", "instruction", "framing",
                "chatter", "rationale",
            ],
            "UNGROUNDED_INJECTION": [
                "hallucinate", "not in source", "unsupported",
                "external", "added fact",
            ],
            "REPETITIVE_CONTENT": [
                "redundant", "again", "repeated", "duplicate", "looping",
            ],
            "GROUNDED_OVERGENERATION": [
                "too detailed", "excessive", "source supported",
                "over-detailed", "beyond scope",
            ],
        }

        self.bert_model_keys = [
            col for col in self.data.columns if col.startswith("bert_")
        ]

    def _bert_flagged_ensemble(self, row, label):
        return float(row.get(f"{label}_prob", 0.0)) >= self.BERT_FLAG_THRESHOLD

    def _bert_flagged_any_individual(self, row, label):
        if not self.bert_model_keys:
            return self._bert_flagged_ensemble(row, label)

        for model_key in self.bert_model_keys:
            model_scores = row.get(model_key, {})
            if isinstance(model_scores, dict):
                prob = float(model_scores.get(label, 0.0))
                if prob >= self.BERT_FLAG_THRESHOLD:
                    return True
        return False

    def _collect_llm_responses(self, row):
        if not row.get("llm_reviewed", False):
            return None

        llm_cols = [
            c for c in self.data.columns
            if c.startswith("llm_") and c != "llm_reviewed"
        ]
        return [
            str(row[col]) for col in llm_cols
            if row.get(col) and row[col] != PLACEHOLDER
        ]

    def _calculate_row_conflict(self, row):
        responses = self._collect_llm_responses(row)

        if responses is None:
            return 0.0

        text = " ".join(responses).lower()
        score = 0.0

        for label, keywords in self.taxonomy_keywords.items():
            bert_detected = self._bert_flagged_any_individual(row, label)
            llm_detected  = any(k in text for k in keywords)

            if not bert_detected and llm_detected:
                score += 0.6
            if bert_detected and not llm_detected:
                score += 0.2

        llm_says_clean = any(
            k in text for k in ["no error", "clean", "correct", "validated"]
        )
        bert_flagged_anything = any(
            self._bert_flagged_any_individual(row, l)
            for l in self.taxonomy_keywords
        )
        if llm_says_clean and bert_flagged_anything:
            score += 0.4

        return score

    def audit(self):
        df = self.data.copy()

        df["conflict_score"] = df.apply(self._calculate_row_conflict, axis=1)
        df["final_decision"] = df.apply(
            lambda row: (
                "SKIPPED"
                if not row.get("llm_reviewed", False)
                else "REVIEW REQUIRED"
                if row["conflict_score"] >= 0.5
                else "VALIDATED"
            ),
            axis=1,
        )

        total     = len(df)
        flagged   = (df["final_decision"] == "REVIEW REQUIRED").sum()
        validated = (df["final_decision"] == "VALIDATED").sum()
        skipped   = (df["final_decision"] == "SKIPPED").sum()

        print(f"Audit complete.")
        print(f"  Total sentences  : {total}")
        print(f"  Review required  : {flagged}  ({100 * flagged / total:.1f}%)")
        print(f"  Validated        : {validated}  ({100 * validated / total:.1f}%)")
        print(f"  Skipped (no LLM) : {skipped}  ({100 * skipped / total:.1f}%)")

        return df