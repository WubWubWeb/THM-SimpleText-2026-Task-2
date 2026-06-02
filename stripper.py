import json


def strip_confident_nones(original_test_path, distilbert_predictions_path, output_filtered_path, none_threshold=0.8):
    with open(original_test_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    with open(distilbert_predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    confident_none_ids = set()

    for pred in predictions:
        if pred.get("None_prob", 0.0) >= none_threshold:
            confident_none_ids.add(pred["snt_id"])

    filtered_docs = []

    for doc in original_data:
        new_sentences = []
        new_labels = []

        for idx, sentence in enumerate(doc["sentences"]):
            snt_id = f"{doc['id']}_{idx}"

            if snt_id not in confident_none_ids:
                new_sentences.append(sentence)

                if "labels" in doc:
                    new_labels.append(doc["labels"][idx])

        if len(new_sentences) > 0:
            new_doc = {
                "id": doc["id"],
                "source": doc["source"],
                "sentences": new_sentences,
            }

            if "labels" in doc:
                new_doc["labels"] = new_labels

            filtered_docs.append(new_doc)

    with open(output_filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered_docs, f, indent=4)

