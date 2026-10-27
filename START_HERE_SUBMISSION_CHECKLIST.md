# START HERE — What to Submit (Kavin Raj SK · 25BCE2866 · BAMAT207)

Everything for **Probabilistic SMS Spam Detection** is done and is in this folder.
Submission deadline on the activity sheet: **27.10.2026**.

## 📤 Submit these (in whatever format your faculty's portal asks)

| # | Deliverable asked for | Your ready file |
|---|---|---|
| 1 | Python / Google Colab notebook | `SMS_Spam_Detection_KavinRajSK_25BCE2866.ipynb` (already executed — outputs visible) + `sms_spam_detection.py` |
| 2 | Cleaned dataset summary | `outputs/cleaned_dataset_summary.csv` (+ Table 2 in the report, `data/cleaned_dataset.csv`) |
| 3 | Class-distribution chart | `figures/class_distribution.png` (+ 4 more charts in `figures/`) |
| 4 | Probabilities for 5 suspicious words | `outputs/word_probabilities.csv` (+ Table 4, Bayes check Table 5 in report) |
| 5 | Trained Naive Bayes model | `model/multinomial_nb_model.joblib` + `model/count_vectorizer.joblib` |
| 6 | Confusion matrix & performance report | `figures/confusion_matrix.png`, `outputs/model_performance.csv` (+ Tables 11–14 in report) |
| 7 | GitHub repo link | Follow `GITHUB_UPLOAD_GUIDE.md` (5 min, needs your own login) then paste the link at submission |
| ★ | Project report | `BAMAT207_Probabilistic_SMS_Spam_Detection_KavinRajSK_25BCE2866.pdf` (submit this) · `.docx` (editable backup) |

## ✅ Before you submit — 2-minute checks
- [ ] Open the PDF. Your name **Kavin Raj SK**, Reg No **25BCE2866**, **B.TECH CSE--CORE** are on the cover. ✔ (already filled)
- [ ] Open the notebook → *Run All* once yourself, so you know it works on your machine/Colab.
- [ ] Upload to GitHub (guide provided) and keep the link ready.
- [ ] Read `VIVA_PREPARATION.md` — 15 likely questions with your exact numbers as answers.
- [ ] If your college needs a specific cover-page format (college name/logo, faculty name), open the `.docx`, add those two lines on page 1, and re-export as PDF. Everything else stays the same.

## 🔑 Your 5 numbers to remember for the viva
1. Cleaned data: **5,160** messages (4,518 ham, 642 spam) after removing **414** duplicates → **P(Spam) = 0.1244**
2. **P(Spam \| Free) = 0.7236** (144/199) — proved by Bayes' theorem; *prize* is the strongest word (70/70 = 1.0)
3. **E(Y) = 0.1244, Var(Y) = 0.1089** because Y is Bernoulli: E(Y)=p, Var(Y)=p(1−p)
4. Model: **98.84% accuracy**, spam precision **98.33%**, recall **92.19%**, F1 **95.16%**, confusion matrix **[[902, 2], [10, 118]]**
5. *"Congratulations! You have won a free prize. Claim now."* → **Spam, P = 0.999999641**

## 💡 Why your numbers differ from classmates (a good thing)
Your split uses `random_state = 2866` (last four digits of YOUR Reg No.), so your test set, vocabulary (7,597), metrics and confusion matrix are your own reproducible run — explain it exactly like that if asked.
