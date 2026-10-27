"""
Probabilistic SMS Spam Detection
BAMAT207 - Probability and Statistics (Project Activity)

Author          : Kavin Raj SK
Registration No.: 25BCE2866
Programme       : B.Tech, CSE - Core

What this program does
----------------------
1. Loads the UCI SMS Spam Collection (dataset ID 228) with ``ucimlrepo``
   and falls back to the local UCI ZIP file when there is no internet.
2. Cleans the corpus (missing-value check, whitespace trimming,
   duplicate removal) and encodes the labels as Spam = 1, Ham = 0.
3. Computes the empirical prior probabilities P(Spam) and P(Ham).
4. Computes word-level conditional probabilities P(Spam | Word) for the
   five suspicious words named in the activity sheet
   (free, prize, win, urgent, offer) and verifies P(Spam | Free)
   with Bayes' theorem.
5. Treats the label as a Bernoulli random variable Y and reports
   E(Y) and Var(Y).
6. Trains a Multinomial Naive Bayes classifier on CountVectorizer
   word-count features using a stratified 80:20 split.
7. Evaluates the classifier (accuracy, precision, recall, F1-score,
   confusion matrix, classification report) and inspects its mistakes.
8. Scores new SMS messages in real time and prints
   P(Spam | Message) and P(Ham | Message).

Design choice - random seed
---------------------------
RANDOM_STATE is set to 2866, the last four digits of my registration
number (25BCE2866). Any fixed seed makes the split reproducible;
using my own number simply makes this run unmistakably mine.

Run:
    python sms_spam_detection.py              # full analysis + demo tests
    python sms_spam_detection.py --interactive  # then type your own SMS
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures even without a display (Colab-safe too)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 2866          # last four digits of Reg. No. 25BCE2866
TEST_SIZE = 0.20             # 80% train / 20% test, as specified
TARGET_WORDS = ["free", "prize", "win", "urgent", "offer"]   # from activity sheet
BONUS_WORDS = ["claim", "call", "text"]                      # my own extra check
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

# Chart palette (kept identical in the written report)
COLOUR_HAM = "#2A9D8F"       # teal  -> genuine messages
COLOUR_SPAM = "#E76F51"      # burnt orange -> spam
COLOUR_DARK = "#264653"      # deep slate for titles/annotations

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.dpi": 200,
})


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------
def _read_zip_member(zip_path: Path) -> pd.DataFrame:
    """Read the tab-separated SMSSpamCollection file inside the UCI ZIP."""
    with zipfile.ZipFile(zip_path) as archive:
        member = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
        raw_bytes = archive.read(member)
    return pd.read_csv(
        io.BytesIO(raw_bytes), sep="\t", header=None,
        names=["label", "message"], quoting=3, encoding="utf-8",
    )


def load_sms_dataset() -> tuple[pd.DataFrame, str]:
    """Try the UCI API first; fall back to a local ZIP so the work is offline-safe."""
    try:
        from ucimlrepo import fetch_ucirepo

        fetched = fetch_ucirepo(id=228)
        frame = pd.concat([fetched.data.features, fetched.data.targets], axis=1)
        # The API column names can vary, so locate the label column by its values.
        label_col = None
        for col in frame.columns:
            values = set(frame[col].astype(str).str.lower().unique())
            if values.issubset({"ham", "spam"}) and len(values) == 2:
                label_col = col
                break
        if label_col is None:
            raise ValueError("Could not identify the ham/spam column from the API response.")
        text_col = next(c for c in frame.columns if c != label_col)
        data = frame[[label_col, text_col]].copy()
        data.columns = ["label", "message"]
        return data, "UCI Machine Learning Repository via ucimlrepo (dataset ID 228)"
    except Exception as api_error:  # noqa: BLE001 - any failure should trigger the fallback
        print(f"[info] Online UCI fetch unavailable ({api_error}). Using local ZIP instead.")

    candidates = [
        DATA_DIR / "sms+spam+collection.zip",
        BASE_DIR / "sms+spam+collection.zip",
        Path.cwd() / "sms+spam+collection.zip",
        Path.cwd() / "data" / "sms+spam+collection.zip",
        Path("/content/sms+spam+collection.zip"),  # Google Colab working folder
    ]
    for candidate in candidates:
        if candidate.exists():
            return _read_zip_member(candidate), f"Local UCI ZIP file: {candidate}"
    raise FileNotFoundError(
        "Dataset not found. Install ucimlrepo with internet access, or place "
        "'sms+spam+collection.zip' in the data/ folder next to this script."
    )


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise labels/text, drop unusable rows and repeated messages, encode labels."""
    data = raw.copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"] != ""]
    data = data.drop_duplicates(subset=["message"], keep="first")
    data = data.reset_index(drop=True)
    data["label_encoded"] = data["label"].map({"ham": 0, "spam": 1}).astype(int)
    return data


# ---------------------------------------------------------------------------
# 3. Word-level probability analysis
# ---------------------------------------------------------------------------
def contains_whole_word(series: pd.Series, word: str) -> pd.Series:
    """True where the complete word occurs at least once (case-insensitive)."""
    pattern = rf"\b{re.escape(word)}\b"
    return series.str.contains(pattern, case=False, regex=True)


def word_probability_table(data: pd.DataFrame, words: list[str]) -> pd.DataFrame:
    """Count-based conditional probabilities for each suspicious word."""
    total = len(data)
    n_spam = int((data["label"] == "spam").sum())
    n_ham = int((data["label"] == "ham").sum())
    p_spam = n_spam / total
    rows = []
    for word in words:
        has_word = contains_whole_word(data["message"], word)
        with_word = int(has_word.sum())
        spam_with = int((has_word & (data["label"] == "spam")).sum())
        ham_with = int(with_word - spam_with)
        p_spam_given_word = spam_with / with_word if with_word else float("nan")
        rows.append({
            "Word": word,
            "Messages containing word": with_word,
            "Spam messages": spam_with,
            "Ham messages": ham_with,
            "P(Word)": with_word / total,
            "P(Word | Spam)": spam_with / n_spam,
            "P(Word | Ham)": ham_with / n_ham,
            "P(Spam | Word)": p_spam_given_word,
            "P(Ham | Word)": ham_with / with_word if with_word else float("nan"),
            # Lift: how many times more likely spam becomes once the word is seen.
            "Lift over prior": p_spam_given_word / p_spam if with_word else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def save_class_distribution(data: pd.DataFrame) -> Path:
    counts = data["label"].value_counts().reindex(["ham", "spam"])
    total = len(data)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(
        ["Ham\n(genuine)", "Spam\n(unwanted)"],
        [counts["ham"], counts["spam"]],
        color=[COLOUR_HAM, COLOUR_SPAM], edgecolor="white", width=0.55,
    )
    for bar, label in zip(bars, ["ham", "spam"]):
        value = int(counts[label])
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.012,
                f"{value:,}\n({value / total:.2%})", ha="center", va="bottom",
                fontsize=10, color=COLOUR_DARK, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("Number of messages")
    ax.set_title(f"Class distribution after cleaning (N = {total:,})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = FIGURE_DIR / "class_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_word_chart(table: pd.DataFrame) -> Path:
    ordered = table.sort_values("P(Spam | Word)")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.barh(ordered["Word"], ordered["P(Spam | Word)"],
                   color=COLOUR_SPAM, edgecolor="white", height=0.58)
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(min(bar.get_width() + 0.02, 1.02), bar.get_y() + bar.get_height() / 2,
                f"{row['P(Spam | Word)']:.2%}  ({int(row['Spam messages'])}/"
                f"{int(row['Messages containing word'])})",
                va="center", fontsize=9, color=COLOUR_DARK)
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("P(Spam | Word)  —  share of messages containing the word that are spam")
    ax.set_title("How strongly does each suspicious word point to spam?")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = FIGURE_DIR / "word_spam_probability.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_length_chart(data: pd.DataFrame) -> Path:
    lengths = data.copy()
    lengths["chars"] = lengths["message"].str.len()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bins = np.arange(0, 325, 15)
    for label, colour, name in [("ham", COLOUR_HAM, "Ham"), ("spam", COLOUR_SPAM, "Spam")]:
        subset = lengths.loc[lengths["label"] == label, "chars"].clip(upper=320)
        ax.hist(subset, bins=bins, alpha=0.62, color=colour, label=name, edgecolor="white")
        ax.axvline(subset.median(), color=colour, linestyle="--", linewidth=1.6)
    ax.set_xlabel("Message length (characters; values above 320 grouped at the edge)")
    ax.set_ylabel("Number of messages")
    ax.set_title("Message length by class (dashed lines = class medians)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = FIGURE_DIR / "message_length_by_class.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion_matrix(cm: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(cm, cmap="YlOrBr", vmin=0)
    ax.set_xticks([0, 1], labels=["Predicted Ham", "Predicted Spam"])
    ax.set_yticks([0, 1], labels=["Actual Ham", "Actual Spam"])
    tags = [["TN", "FP"], ["FN", "TP"]]
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row - 0.12, tags[row][col], ha="center", va="center",
                color="white" if value > cm.max() * 0.55 else COLOUR_DARK,
                fontsize=10, fontweight="bold")
        ax.text(col, row + 0.16, f"{int(value):,}", ha="center", va="center",
                color="white" if value > cm.max() * 0.55 else COLOUR_DARK, fontsize=13)
    ax.set_title("Confusion matrix — held-out test set")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIGURE_DIR / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Model training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(data: pd.DataFrame) -> dict:
    X_text = data["message"]
    y = data["label_encoded"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    # Fit the vocabulary on training messages ONLY (prevents test-set leakage).
    vectorizer = CountVectorizer(lowercase=True)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    ham_col = list(model.classes_).index(0)
    spam_col = list(model.classes_).index(1)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Spam precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "Spam recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "Spam F1-score": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "Ham precision": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "Ham recall": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "Ham F1-score": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report_text = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Ham (0)", "Spam (1)"], digits=4, zero_division=0,
    )
    report_dict = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Ham (0)", "Spam (1)"], digits=6, zero_division=0, output_dict=True,
    )
    errors = pd.DataFrame({
        "message": X_test_text.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": y_pred,
        "P(Ham)": probabilities[:, ham_col],
        "P(Spam)": probabilities[:, spam_col],
    }, index=X_test_text.index)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "X_train_text": X_train_text, "X_test_text": X_test_text,
        "y_train": y_train, "y_test": y_test, "y_pred": y_pred,
        "metrics": metrics, "confusion_matrix": cm,
        "classification_report": report_text, "report_dict": report_dict,
        "errors": errors,
        "false_negatives": errors[(errors["actual"] == 1) & (errors["predicted"] == 0)],
        "false_positives": errors[(errors["actual"] == 0) & (errors["predicted"] == 1)],
        "vocabulary_size": len(vectorizer.get_feature_names_out()),
    }


# ---------------------------------------------------------------------------
# 6. Real-time prediction
# ---------------------------------------------------------------------------
def predict_sms(message: str, vectorizer: CountVectorizer, model: MultinomialNB) -> dict:
    """Score one new SMS with the already-fitted pipeline and print both probabilities."""
    features = vectorizer.transform([message])
    probs = model.predict_proba(features)[0]
    by_class = {int(c): float(p) for c, p in zip(model.classes_, probs)}
    predicted = int(model.predict(features)[0])
    label = "Spam" if predicted == 1 else "Ham (genuine)"
    print(f"\nSMS: {message}")
    print(f"Prediction: {label}")
    print(f"P(Spam | Message) = {by_class[1]:.9f}")
    print(f"P(Ham | Message)  = {by_class[0]:.9f}")
    return {"message": message, "prediction": label,
            "P(Spam | Message)": by_class[1], "P(Ham | Message)": by_class[0]}


def interactive_test(vectorizer: CountVectorizer, model: MultinomialNB) -> None:
    print("\nType an SMS to test it. Enter 'exit' or 'quit' to stop.")
    while True:
        try:
            message = input("Enter an SMS message: ")
        except EOFError:
            break
        if message.strip().lower() in {"exit", "quit"}:
            break
        if not message.strip():
            print("Please enter a non-empty message.")
            continue
        predict_sms(message, vectorizer, model)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Probabilistic SMS spam detection")
    parser.add_argument("--interactive", action="store_true",
                        help="after the demo messages, prompt for your own SMS text")
    args = parser.parse_args()

    for folder in (DATA_DIR, FIGURE_DIR, MODEL_DIR, OUTPUT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    # --- Activity 1: dataset handling -----------------------------------
    raw, source = load_sms_dataset()
    print("=" * 72)
    print("PROBABILISTIC SMS SPAM DETECTION - Kavin Raj SK (25BCE2866)")
    print("=" * 72)
    print(f"Dataset source : {source}")
    print(f"Raw shape      : {raw.shape}")
    print("Missing values in the raw data:")
    print(raw.isna().sum().to_string())
    raw_counts = raw["label"].astype(str).str.strip().str.lower().value_counts()

    data = clean_dataset(raw)
    counts = data["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
    total = len(data)
    duplicates_removed = len(raw) - total
    print(f"\nCleaned shape  : {data.shape}  (duplicates removed: {duplicates_removed})")
    print(f"Duplicates left: {int(data.duplicated(subset=['message']).sum())}")
    print(counts.to_string())

    # --- Activity 2: prior probabilities --------------------------------
    p_ham = counts["ham"] / total
    p_spam = counts["spam"] / total
    print(f"\nP(Ham)  = {counts['ham']}/{total} = {p_ham:.7f} ({p_ham:.2%})")
    print(f"P(Spam) = {counts['spam']}/{total} = {p_spam:.7f} ({p_spam:.2%})")
    save_class_distribution(data)

    # Extra descriptive statistics: message length
    data = data.assign(
        char_length=data["message"].str.len(),
        word_count=data["message"].str.split().str.len(),
    )
    length_summary = (
        data.groupby("label")[["char_length", "word_count"]]
        .agg(["mean", "median", "min", "max"]).round(2)
    )
    print("\nMessage-length summary by class:")
    print(length_summary.to_string())
    save_length_chart(data)

    # --- Activity 3: word-based probabilities + Bayes -------------------
    word_table = word_probability_table(data, TARGET_WORDS)
    print("\nWord-based probability table (five suspicious words):")
    print(word_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    save_word_chart(word_table)

    bonus_table = word_probability_table(data, BONUS_WORDS)
    print("\nBonus words (my own extension - for contrast):")
    print(bonus_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    free = word_table.loc[word_table["Word"] == "free"].iloc[0]
    bayes_result = (free["P(Word | Spam)"] * p_spam) / free["P(Word)"]
    direct_result = free["Spam messages"] / free["Messages containing word"]
    print("\nBayes' theorem verification for the word 'free':")
    print(f"  P(Free | Spam) = {free['P(Word | Spam)']:.7f}")
    print(f"  P(Spam)        = {p_spam:.7f}")
    print(f"  P(Free)        = {free['P(Word)']:.7f}")
    print(f"  Bayes  P(Spam | Free) = {bayes_result:.7f}")
    print(f"  Direct count check    = {int(free['Spam messages'])}/"
          f"{int(free['Messages containing word'])} = {direct_result:.7f}")
    assert abs(bayes_result - direct_result) < 1e-9, "Bayes check failed!"

    # --- Activity 4: random variable Y ----------------------------------
    y_values = data["label_encoded"].to_numpy(dtype=float)
    expectation = float(np.mean(y_values))
    variance_population = float(np.var(y_values, ddof=0))
    print("\nRandom variable Y (1 = Spam, 0 = Ham):")
    print(f"  E(Y)   = P(Spam) = {expectation:.7f}")
    print(f"  Var(Y) = p(1-p)  = {expectation * (1 - expectation):.7f}")
    print(f"  NumPy population variance check = {variance_population:.7f}")
    print(f"  SD(Y)  = {np.sqrt(variance_population):.7f}")

    # --- Activities 5-7: split, train, evaluate -------------------------
    fitted = train_and_evaluate(data)
    cm = fitted["confusion_matrix"]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    print(f"\nStratified split (random_state={RANDOM_STATE}): "
          f"train = {len(fitted['X_train_text'])}, test = {len(fitted['X_test_text'])}")
    print(f"Vocabulary size (training text only): {fitted['vocabulary_size']:,}")
    print("\nModel performance on the held-out test set:")
    for name, value in fitted["metrics"].items():
        print(f"  {name:<16} {value:.6f}  ({value:.2%})")
    print(f"\nConfusion matrix [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    print("\nClassification report:")
    print(fitted["classification_report"])
    save_confusion_matrix(cm)

    print(f"False negatives - spam let through as ham: {fn}")
    print(fitted["false_negatives"][["message", "P(Ham)", "P(Spam)"]].head(5).to_string(index=False))
    print(f"\nFalse positives - genuine messages flagged as spam: {fp}")
    print(fitted["false_positives"][["message", "P(Ham)", "P(Spam)"]].head(5).to_string(index=False))

    # --- Activity 8: real-time testing ----------------------------------
    demo_messages = [
        "Congratulations! You have won a free prize. Claim now.",   # activity-sheet example
        "URGENT! Your account will be blocked. Verify your details now to claim your offer.",
        "Hi, are we still meeting for lunch tomorrow at 1?",
        "Your OTP is 482913. Please do not share it with anyone.",
        "WINNER! You have been selected for a free holiday. Call now to claim your prize!",
    ]
    print("\n" + "-" * 72)
    print("REAL-TIME SMS TESTS")
    print("-" * 72)
    demo_results = [predict_sms(m, fitted["vectorizer"], fitted["model"]) for m in demo_messages]

    # --- Save every deliverable artefact --------------------------------
    data[["label", "message", "label_encoded"]].to_csv(DATA_DIR / "cleaned_dataset.csv", index=False)
    word_table.to_csv(OUTPUT_DIR / "word_probabilities.csv", index=False)
    bonus_table.to_csv(OUTPUT_DIR / "bonus_word_probabilities.csv", index=False)
    pd.DataFrame(demo_results).to_csv(OUTPUT_DIR / "realtime_test_results.csv", index=False)
    fitted["false_negatives"].to_csv(OUTPUT_DIR / "false_negatives.csv")
    fitted["false_positives"].to_csv(OUTPUT_DIR / "false_positives.csv")
    joblib.dump(fitted["vectorizer"], MODEL_DIR / "count_vectorizer.joblib")
    joblib.dump(fitted["model"], MODEL_DIR / "multinomial_nb_model.joblib")

    length_flat = {}
    for label in ["ham", "spam"]:
        subset = data[data["label"] == label]
        length_flat[label] = {
            "char_mean": float(subset["char_length"].mean()),
            "char_median": float(subset["char_length"].median()),
            "char_min": int(subset["char_length"].min()),
            "char_max": int(subset["char_length"].max()),
            "word_mean": float(subset["word_count"].mean()),
            "word_median": float(subset["word_count"].median()),
        }

    results = {
        "author": "Kavin Raj SK",
        "registration_number": "25BCE2866",
        "dataset_source": source,
        "random_state": RANDOM_STATE,
        "raw_rows": int(len(raw)),
        "raw_ham": int(raw_counts.get("ham", 0)),
        "raw_spam": int(raw_counts.get("spam", 0)),
        "raw_missing_label": int(raw["label"].isna().sum()),
        "raw_missing_message": int(raw["message"].isna().sum()),
        "duplicates_removed": int(duplicates_removed),
        "cleaned_rows": int(total),
        "cleaned_ham": int(counts["ham"]),
        "cleaned_spam": int(counts["spam"]),
        "p_ham": float(p_ham),
        "p_spam": float(p_spam),
        "expectation_Y": expectation,
        "variance_Y": variance_population,
        "sd_Y": float(np.sqrt(variance_population)),
        "word_table": word_table.to_dict(orient="records"),
        "bonus_word_table": bonus_table.to_dict(orient="records"),
        "bayes_free": {
            "p_free_given_spam": float(free["P(Word | Spam)"]),
            "p_spam": float(p_spam),
            "p_free": float(free["P(Word)"]),
            "p_free_and_spam": float(free["Spam messages"] / total),
            "p_spam_given_free_bayes": float(bayes_result),
            "p_spam_given_free_direct": float(direct_result),
        },
        "length_summary": length_flat,
        "train_size": int(len(fitted["X_train_text"])),
        "test_size": int(len(fitted["X_test_text"])),
        "train_ham": int((fitted["y_train"] == 0).sum()),
        "train_spam": int((fitted["y_train"] == 1).sum()),
        "test_ham": int((fitted["y_test"] == 0).sum()),
        "test_spam": int((fitted["y_test"] == 1).sum()),
        "vocabulary_size": int(fitted["vocabulary_size"]),
        "metrics": fitted["metrics"],
        "classification_report": fitted["report_dict"],
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "false_negative_examples": fitted["false_negatives"].head(6).to_dict(orient="records"),
        "false_positive_examples": fitted["false_positives"].head(6).to_dict(orient="records"),
        "demo_results": demo_results,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nArtefacts saved under: {BASE_DIR}")
    print("  data/cleaned_dataset.csv | figures/*.png | model/*.joblib | outputs/*")

    if args.interactive:
        interactive_test(fitted["vectorizer"], fitted["model"])


if __name__ == "__main__":
    main()
