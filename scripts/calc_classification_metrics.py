import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
        raise ValueError(f"Unsupported JSON structure: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_label(value):
    return str(value).strip().lower().replace("_", " ")


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def compute_metrics(y_true, y_pred, minority_classes=None):
    labels = sorted(set(y_true) | set(y_pred))
    support = Counter(y_true)
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    per_class = {}
    for label in labels:
        precision = safe_div(tp[label], tp[label] + fp[label])
        recall = safe_div(tp[label], tp[label] + fn[label])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[label]),
        }

    accuracy = safe_div(sum(1 for t, p in zip(y_true, y_pred) if t == p), len(y_true))
    macro_f1 = safe_div(sum(item["f1"] for item in per_class.values()), len(labels))

    if minority_classes is None:
        nonzero = [count for count in support.values() if count > 0]
        min_support = min(nonzero) if nonzero else 0
        minority_classes = [label for label, count in support.items() if count == min_support]
    else:
        minority_classes = [normalize_label(label) for label in minority_classes]

    minority_recalls = {
        label: per_class.get(label, {"recall": 0.0})["recall"]
        for label in minority_classes
    }
    minority_recall = safe_div(sum(minority_recalls.values()), len(minority_recalls))

    confusion = defaultdict(Counter)
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    return {
        "num_samples": len(y_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "minority_classes": minority_classes,
        "minority_recall": minority_recall,
        "minority_recalls": minority_recalls,
        "per_class": per_class,
        "confusion_matrix": {
            true: {pred: int(count) for pred, count in preds.items()}
            for true, preds in confusion.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate Accuracy, Macro-F1 and minority-class Recall.")
    parser.add_argument("--pred-file", required=True, help="CSV/JSON/JSONL with ground-truth and predicted labels.")
    parser.add_argument("--true-col", default="true_label")
    parser.add_argument("--pred-col", default="pred_label")
    parser.add_argument("--minority-class", action="append", default=None, help="Minority class name. Can be repeated.")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    rows = load_rows(args.pred_file)
    y_true = []
    y_pred = []
    for row in rows:
        if args.true_col not in row or args.pred_col not in row:
            continue
        y_true.append(normalize_label(row[args.true_col]))
        y_pred.append(normalize_label(row[args.pred_col]))

    if not y_true:
        raise ValueError(f"No valid rows found with columns {args.true_col!r} and {args.pred_col!r}")

    results = compute_metrics(y_true, y_pred, args.minority_class)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
