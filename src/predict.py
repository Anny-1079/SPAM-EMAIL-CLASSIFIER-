"""
Classify emails with the trained spam model.

Usage (from the project root):

    python src/predict.py                              # classifies built-in examples
    python src/predict.py "Subject line. Email body"   # classifies the given text
    python src/predict.py --file email.txt             # classifies the contents of a text file
    python src/predict.py --csv emails.csv             # classifies a CSV with Subject and Message columns

The CSV mode writes ``<name>_predictions.csv`` next to the input file.
Run ``python src/spam_classifier.py`` first to train and save the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from text_processing import preprocess

MODEL_PATH = Path(__file__).resolve().parent.parent / "outputs" / "spam_model.joblib"

EXAMPLES = [
    ("Meeting moved to 3pm", "Hi Sarah, can we move tomorrow's project meeting to 3pm? I've attached the updated agenda. Thanks, Mark"),
    ("Cheap meds online!!!", "Buy cheap viagra and cialis online, no prescription needed. Click http://pharma-deals.example now for 80% off, limited time offer!"),
    ("URGENT: claim your prize", "You have been selected to receive $1,000,000. Reply with your bank details to claim your free prize today."),
]


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run: python src/spam_classifier.py")
    return joblib.load(MODEL_PATH)


def classify(model: dict, subjects: list, messages: list) -> pd.DataFrame:
    cleaned = [preprocess(s, m) for s, m in zip(subjects, messages)]
    proba = model["pipeline"].predict_proba(cleaned)[:, 1]
    return pd.DataFrame({
        "spam_probability": proba.round(4),
        "prediction": ["SPAM" if p >= model["threshold"] else "HAM" for p in proba],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Spam email classifier")
    parser.add_argument("text", nargs="?", help="email text (subject and body)")
    parser.add_argument("--file", help="path to a text file containing one email")
    parser.add_argument("--csv", help="path to a CSV with Subject and Message columns")
    args = parser.parse_args()
    model = load_model()
    print(f"Model: {model['model_name']} | spam threshold: {model['threshold']:.3f}\n")

    if args.csv:
        path = Path(args.csv)
        data = pd.read_csv(path)
        result = classify(model, data["Subject"].tolist(), data["Message"].tolist())
        out = path.with_name(path.stem + "_predictions.csv")
        pd.concat([data, result], axis=1).to_csv(out, index=False)
        print(f"Classified {len(data):,} emails ({(result.prediction == 'SPAM').sum():,} spam). Saved to {out}")
        return

    if args.file:
        items = [(None, Path(args.file).read_text(errors="ignore"))]
    elif args.text:
        items = [(None, args.text)]
    else:
        items = EXAMPLES

    result = classify(model, [s for s, _ in items], [m for _, m in items])
    for (subject, body), row in zip(items, result.itertuples()):
        label = f"{subject}: {body}" if subject else body
        print(f"[{row.prediction:<4}] spam probability {row.spam_probability:.3f} | {label[:90].strip()}")


if __name__ == "__main__":
    main()
