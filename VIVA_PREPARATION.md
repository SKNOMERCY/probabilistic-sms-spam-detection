# Viva Preparation — Probabilistic SMS Spam Detection
**Kavin Raj SK · 25BCE2866 · BAMAT207** — likely questions, with answers using *your* numbers. Read this twice and run the notebook once; you'll be fine.

---

**Q1. What is your project, in one line?**
Classify an SMS as Spam (1) or Ham/genuine (0) using probability — priors, conditional probability and Bayes' theorem first, then a Multinomial Naive Bayes model — on the UCI SMS Spam Collection (5,574 messages).

**Q2. What did you do to clean the data?**
Checked missing values (0 in both columns), trimmed whitespace, lower-cased labels, removed **414 duplicate messages** → **5,160 unique** messages (4,518 ham, 642 spam), encoded Spam = 1, Ham = 0. Duplicates are removed *before* splitting so the same text can't appear in both train and test.

**Q3. What are P(Spam) and P(Ham)? How did you get them?**
P(Spam) = 642/5160 = **0.1244 (12.44%)**; P(Ham) = 4518/5160 = **0.8756 (87.56%)**. Empirical probability = class count ÷ total. They sum to 1.

**Q4. What is P(Spam | Free) and how was it computed?**
**0.7236** — of 199 messages containing the whole word *free*, 144 are spam (144/199).

**Q5. Verify it with Bayes' theorem.**
P(Spam|Free) = P(Free|Spam)·P(Spam) / P(Free) = (144/642 × 642/5160) / (199/5160) = (0.2243 × 0.1244) / 0.03857 = **0.7236** — same as the direct count. Bayes just rearranges the same counts.

**Q6. Which of the five words is the strongest spam indicator? Weakest?**
*prize* — 70/70 (**1.0000**) in this corpus; then urgent 0.9180, win 0.8868, offer 0.8000, *free* weakest at 0.7236 (55 ham messages also contain it). Bonus: *claim* is 91/91, but everyday *call*/*text* are only ~0.61/0.59. (Caveat: 1.0 means "no ham example in this sample", not "impossible".)

**Q7. Define your random variable. E(Y)? Var(Y)?**
Y = 1 if spam, 0 if ham → Bernoulli with p = P(Spam) = 0.1244. **E(Y) = p = 0.1244**; **Var(Y) = p(1−p) = 0.1089**; SD = 0.3301. Verified with NumPy mean/variance.

**Q8. How did you split the data, and why stratified?**
80:20 `train_test_split` with `stratify=y` so both parts keep the 87.56:12.44 ham–spam ratio (test = 904 ham, 128 spam). `random_state = 2866` — the last digits of my Reg No. — so the split is reproducible.

**Q9. What does CountVectorizer do? Why fit it only on training data?**
It lower-cases text, builds a vocabulary (**7,597 tokens** here) and turns each SMS into a word-count vector. Fitting it on all data would leak test vocabulary into training and inflate the score — so train `fit_transform`, test only `transform`.

**Q10. Why Multinomial Naive Bayes? What is "naive"?**
It's designed for discrete count features and is Bayes' theorem applied to all words at once: P(class|message) ∝ P(class) × ∏ P(word|class)^count. "Naive" = it assumes words are independent given the class. False in real language, but fast and very effective as a baseline.

**Q11. Your results?**
Accuracy **98.84%**, spam precision **98.33%**, recall **92.19%**, F1 **95.16%**. Confusion matrix **[[902, 2], [10, 118]]** = TN 902, FP 2, FN 10, TP 118.

**Q12. Precision vs recall — which matters here?**
Precision: of messages *flagged* spam, how many really are (98.33% → very few genuine messages hidden). Recall: of *all* spam, how much we catch (92.19% → 10 slip through). Both matter: an FP can hide an OTP; an FN lets a scam through. That's also why accuracy alone misleads on 87.6%-ham data — always saying "ham" scores 87.56% while catching zero spam.

**Q13. Show a real-time prediction.**
"Congratulations! You have won a free prize. Claim now." → **Spam, P(Spam|Msg) = 0.999999641**. (The activity sheet's 0.96/0.04 is just a format example — mine are the computed values.) An OTP message correctly scores Ham (P Spam ≈ 0.0002).

**Q14. Where does your model fail?**
**2 false positives** (a formal job message; the 3-word "Waiting for your call.") and **10 false negatives** — spam written like personal chat/receipts or with obfuscated tokens like `PICSFREE1`. Bag-of-words ignores word order, sender, links and context.

**Q15. Limitations / future work?**
Historical English corpus (new spam vocabulary, links, other languages); one fixed split (use stratified cross-validation for variability); Naive Bayes probabilities are over-confident (calibrate before risk decisions); character n-grams for misspellings; compare with logistic regression/SVM; validate on a recent corpus.

---
**If asked something you don't know:** say *"In my implementation I used the scikit-learn default for that (alpha = 1.0 Laplace smoothing / CountVectorizer default token pattern), and tuned nothing — the report's Section 9.2 lists it as future work."* Honest + correct.
