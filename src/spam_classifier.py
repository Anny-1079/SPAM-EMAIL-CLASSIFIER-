"""
Spam Email Classifier - end-to-end pipeline
===========================================

Steps
-----
1. Load a labeled email dataset (Enron-Spam: spam vs. legitimate "ham" emails).
2. Clean and preprocess the text (tokenization, stopword removal, stemming...).
3. Convert text to numbers with TF-IDF (unigrams and bigrams).
4. Train Naive Bayes and Logistic Regression classifiers.
5. Evaluate with accuracy, precision and recall. Because a false positive
   (a real email sent to the spam folder) is worse than a missed spam, the
   decision threshold is tuned to reach a precision target.

Run from the project root:

    python src/spam_classifier.py

Figures, metrics and the trained model are written to ``outputs/``.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from text_processing import combine, preprocess_frame

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "enron_spam_data.zip"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 42
PRECISION_TARGET = 0.995  # at most ~1 in 200 emails flagged as spam may be legitimate
SPAM, HAM = 1, 0
C_HAM, C_SPAM = "#2a6f97", "#c1121f"

sns.set_theme(style="whitegrid", context="notebook")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)  # pandas reads the single CSV inside the zip directly
    df["label"] = df["Spam/Ham"].map({"spam": SPAM, "ham": HAM})
    print(f"Loaded {path.name}: {len(df):,} emails, columns: {list(df.columns[:-1])}")
    print("Labels:", df["Spam/Ham"].value_counts().to_dict())
    print("Missing subject:", int(df["Subject"].isna().sum()), "| missing body:", int(df["Message"].isna().sum()))
    return df


# --------------------------------------------------------------------------
# 2. Clean + preprocess
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"rows_raw": len(df)}

    # a) exact duplicates (same subject, body and label)
    n = len(df)
    df = df.drop_duplicates(subset=["Subject", "Message", "label"])
    log["exact_duplicates_removed"] = n - len(df)

    # b) emails with no text at all
    n = len(df)
    df = df[df.apply(lambda r: bool(combine(r["Subject"], r["Message"]).strip()), axis=1)]
    log["empty_emails_removed"] = n - len(df)

    # c) text preprocessing: tokenize, remove stopwords, stem (see text_processing.py)
    t0 = time.time()
    df = df.copy()
    df["clean_text"] = preprocess_frame(df)
    print(f"Preprocessed {len(df):,} emails in {time.time() - t0:.0f}s")

    n = len(df)
    df = df[df["clean_text"].str.len() > 0]
    log["empty_after_preprocessing_removed"] = n - len(df)

    # d) near-duplicates: different raw text, identical processed text. If these are
    #    split across train and test, the model is graded on emails it has already seen.
    n = len(df)
    labels_per_text = df.groupby("clean_text")["label"].nunique()
    conflicting = labels_per_text[labels_per_text > 1].index
    df = df[~df["clean_text"].isin(conflicting)]  # same text labelled both ways: unusable
    log["conflicting_label_rows_removed"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates(subset="clean_text")
    log["near_duplicates_removed"] = n - len(df)

    df["n_tokens"] = df["clean_text"].str.split().str.len()
    df = df.reset_index(drop=True)
    log["rows_clean"] = len(df)
    log["spam_clean"] = int((df["label"] == SPAM).sum())
    log["ham_clean"] = int((df["label"] == HAM).sum())

    print("Cleaning summary:")
    for k, v in log.items():
        print(f"  {k:<38}{v:>8,}")
    return df, log


# --------------------------------------------------------------------------
# Exploration
# --------------------------------------------------------------------------
def explore(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    counts = df["label"].map({HAM: "Ham (legitimate)", SPAM: "Spam"}).value_counts()
    bars = ax[0].bar(counts.index, counts.values, color=[C_HAM if "Ham" in i else C_SPAM for i in counts.index])
    ax[0].bar_label(bars, fmt="{:,.0f}")
    ax[0].set_title("Class balance after cleaning")
    ax[0].set_ylabel("Emails")
    for label, color, name in [(HAM, C_HAM, "Ham"), (SPAM, C_SPAM, "Spam")]:
        ax[1].hist(df.loc[df["label"] == label, "n_tokens"].clip(upper=600), bins=60,
                   alpha=0.6, color=color, label=name)
    ax[1].set_title("Email length after preprocessing (tokens, capped at 600)")
    ax[1].set_xlabel("Number of tokens")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_balance_and_length.png", dpi=150)
    plt.close(fig)

    # most common words per class (share of emails of that class containing the word)
    cv = CountVectorizer(binary=True, min_df=20)
    X = cv.fit_transform(df["clean_text"])
    vocab = np.array(cv.get_feature_names_out())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for a, (label, color, name) in zip(ax, [(HAM, C_HAM, "ham"), (SPAM, C_SPAM, "spam")]):
        mask = (df["label"] == label).to_numpy()
        share = np.asarray(X[mask].mean(axis=0)).ravel()
        top = np.argsort(share)[-15:]
        a.barh(vocab[top], share[top] * 100, color=color)
        a.set_title(f"Most common words in {name} emails")
        a.set_xlabel("% of emails containing the word")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_common_words_by_class.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3-4. Features + models
# --------------------------------------------------------------------------
def make_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.9, max_features=100_000)),
        ("clf", classifier),
    ])


def tune_models(X_train: pd.Series, y_train: pd.Series) -> dict[str, GridSearchCV]:
    ngrams = [(1, 1), (1, 2)]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    searches = {
        "Naive Bayes": GridSearchCV(
            make_pipeline(MultinomialNB()),
            {"tfidf__ngram_range": ngrams, "clf__alpha": [0.001, 0.003, 0.01, 0.03, 0.1]},
            scoring="average_precision", cv=cv, n_jobs=2, refit=True,
        ),
        "Logistic Regression": GridSearchCV(
            make_pipeline(LogisticRegression(solver="liblinear", max_iter=1000)),
            {"tfidf__ngram_range": ngrams, "clf__C": [10, 30, 100, 300, 1000]},
            scoring="average_precision", cv=cv, n_jobs=2, refit=True,
        ),
    }
    for name, search in searches.items():
        t0 = time.time()
        search.fit(X_train, y_train)
        params = {k.split("__")[1]: v for k, v in search.best_params_.items()}
        print(f"  {name:<20} CV average precision {search.best_score_:.4f} | best: {params} [{time.time() - t0:.0f}s]")
    return searches


# --------------------------------------------------------------------------
# 5. Evaluate
# --------------------------------------------------------------------------
def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[HAM, SPAM]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "true_neg": int(tn), "false_pos": int(fp), "false_neg": int(fn), "true_pos": int(tp),
        "false_positive_rate": fp / (fp + tn),
    }


def choose_threshold(y_true, proba, precision_target: float) -> tuple[float, str]:
    """Lowest threshold whose precision reaches the target (= most spam caught
    while keeping false positives under control). Chosen on out-of-fold
    predictions from the TRAINING data only, never on the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(precision[:-1] >= precision_target)[0]
    if len(ok):
        return float(thresholds[ok[0]]), f"lowest threshold with training precision >= {precision_target:.1%}"
    best = int(np.argmax(precision[:-1]))
    return float(thresholds[best]), f"target {precision_target:.1%} unreachable; used highest-precision threshold"


def plot_confusion(ax, m: dict, title: str) -> None:
    cm = np.array([[m["true_neg"], m["false_pos"]], [m["false_neg"], m["true_pos"]]])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted ham", "Predicted spam"], yticklabels=["Actual ham", "Actual spam"])
    ax.set_title(title)


def evaluation_plots(searches, y_test, probas, best_name, m_default, m_tuned) -> None:
    # confusion matrices: default threshold vs tuned threshold
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    plot_confusion(ax[0], m_default, f"{best_name}, threshold 0.50\n"
                   f"precision {m_default['precision']:.3f} | recall {m_default['recall']:.3f}")
    plot_confusion(ax[1], m_tuned, f"{best_name}, tuned threshold {m_tuned['threshold']:.2f}\n"
                   f"precision {m_tuned['precision']:.3f} | recall {m_tuned['recall']:.3f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ROC and precision-recall curves for both models
    colors = {"Naive Bayes": "#e09f3e", "Logistic Regression": C_HAM}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax[0].plot(fpr, tpr, color=colors[name], label=f"{name} (AUC {roc_auc_score(y_test, proba):.4f})")
        p, r, _ = precision_recall_curve(y_test, proba)
        ax[1].plot(r, p, color=colors[name], label=f"{name} (AP {average_precision_score(y_test, proba):.4f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    ax[1].set(xlabel="Recall (spam caught)", ylabel="Precision (flagged mail that is spam)",
              title="Precision-recall curve", ylim=(0.9, 1.005))
    ax[1].scatter([m_tuned["recall"]], [m_tuned["precision"]], color=C_SPAM, zorder=5, s=60,
                  label=f"Tuned threshold ({best_name})")
    for a in ax:
        a.legend(loc="lower right" if a is ax[0] else "lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_roc_and_precision_recall.png", dpi=150)
    plt.close(fig)


def top_words(lr_pipe: Pipeline, n: int = 20) -> pd.DataFrame:
    """Words (stems) that push Logistic Regression most strongly toward spam or ham."""
    words = np.array(lr_pipe.named_steps["tfidf"].get_feature_names_out())
    coef = lr_pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    out = pd.concat([
        pd.DataFrame({"word": words[order[-n:]][::-1], "coefficient": coef[order[-n:]][::-1], "indicates": "spam"}),
        pd.DataFrame({"word": words[order[:n]], "coefficient": coef[order[:n]], "indicates": "ham"}),
    ], ignore_index=True)
    out["coefficient"] = out["coefficient"].round(3)
    out.to_csv(OUT_DIR / "top_words.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    spam = out[out["indicates"] == "spam"].iloc[::-1]
    ham = out[out["indicates"] == "ham"].iloc[::-1]
    ax[0].barh(spam["word"], spam["coefficient"], color=C_SPAM)
    ax[0].set_title(f"Top {n} words indicating SPAM")
    ax[1].barh(ham["word"], -ham["coefficient"], color=C_HAM)
    ax[1].set_title(f"Top {n} words indicating HAM")
    for a in ax:
        a.set_xlabel("Logistic Regression weight (absolute)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_top_words.png", dpi=150)
    plt.close(fig)
    return out


def snippet(row: pd.Series, n: int = 220) -> str:
    text = " ".join(combine(row["Subject"], row["Message"]).split())
    return text[:n] + ("..." if len(text) > n else "")


def error_analysis(df_test: pd.DataFrame, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    """Save the emails the final model gets wrong, to see WHY it fails."""
    d = df_test.copy()
    d["spam_probability"] = proba.round(4)
    d["predicted"] = (proba >= threshold).astype(int)
    wrong = d[d["predicted"] != d["label"]].copy()
    wrong["error_type"] = np.where(wrong["label"] == HAM, "false positive (ham flagged as spam)",
                                   "false negative (spam missed)")
    wrong["text_start"] = wrong.apply(snippet, axis=1)
    out = wrong.sort_values(["error_type", "spam_probability"], ascending=[True, False])[
        ["error_type", "spam_probability", "text_start"]
    ]
    out.to_csv(OUT_DIR / "misclassified_emails.csv", index=False)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    section("1. LOAD DATA")
    raw = load_data(DATA_PATH)

    section("2. CLEAN + PREPROCESS TEXT")
    df, clean_log = clean_data(raw)
    explore(df)

    section("3. SPLIT")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE)
    X_train, y_train = train_df["clean_text"], train_df["label"]
    X_test, y_test = test_df["clean_text"], test_df["label"]
    print(f"Train: {len(train_df):,} emails ({y_train.mean():.1%} spam) | "
          f"Test: {len(test_df):,} emails ({y_test.mean():.1%} spam)")

    section("4. TF-IDF FEATURES + MODEL TUNING (3-fold CV on training data)")
    searches = tune_models(X_train, y_train)
    probas = {name: s.predict_proba(X_test)[:, 1] for name, s in searches.items()}

    best_name = max(searches, key=lambda k: searches[k].best_score_)  # chosen on CV, not on test
    best_pipe = searches[best_name].best_estimator_
    print(f"\nBest model by cross-validation: {best_name}")
    print(f"TF-IDF vocabulary size: {len(best_pipe.named_steps['tfidf'].vocabulary_):,} features")

    section("5. THRESHOLD TUNING FOR HIGH PRECISION (training data only)")
    oof = cross_val_predict(
        best_pipe, X_train, y_train, method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), n_jobs=2,
    )[:, 1]
    threshold, how = choose_threshold(y_train, oof, PRECISION_TARGET)
    print(f"Chosen threshold: {threshold:.4f} ({how})")

    section("6. EVALUATE ON THE HELD-OUT TEST SET")
    rows = {f"{name} (threshold 0.50)": metrics_at(y_test, p, 0.5) for name, p in probas.items()}
    m_default = metrics_at(y_test, probas[best_name], 0.5)
    m_tuned = metrics_at(y_test, probas[best_name], threshold)
    rows[f"{best_name} (tuned threshold {threshold:.2f})"] = m_tuned
    results = pd.DataFrame(rows).T
    show = results[["accuracy", "precision", "recall", "f1", "roc_auc", "false_pos", "false_neg"]]
    print(show.astype({"false_pos": int, "false_neg": int}).round(4).to_string())
    results.round(4).to_csv(OUT_DIR / "model_comparison.csv", index_label="model")

    evaluation_plots(searches, y_test, probas, best_name, m_default, m_tuned)
    words = top_words(searches["Logistic Regression"].best_estimator_)
    print("\nTop spam words:", ", ".join(words[words.indicates == "spam"].word.head(12)))
    print("Top ham words :", ", ".join(words[words.indicates == "ham"].word.head(12)))
    errors = error_analysis(test_df, probas[best_name], threshold)
    print(f"\nMisclassified test emails saved: {errors['error_type'].value_counts().to_dict()}")

    # persist everything needed to classify new emails
    joblib.dump(
        {"pipeline": best_pipe, "threshold": threshold, "model_name": best_name},
        OUT_DIR / "spam_model.joblib", compress=3,
    )
    metrics = {
        "dataset": "Enron-Spam (Metsis et al., 2006)",
        "cleaning": clean_log,
        "train_size": len(train_df), "test_size": len(test_df),
        "best_model": best_name,
        "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in searches[best_name].best_params_.items()},
        "precision_target": PRECISION_TARGET,
        "threshold_rule": how,
        "cv_average_precision": {n: round(s.best_score_, 4) for n, s in searches.items()},
        "test_default_threshold": m_default,
        "test_tuned_threshold": m_tuned,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nSaved model, metrics and figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
