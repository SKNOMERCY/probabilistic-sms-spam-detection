# GitHub Upload Guide — Kavin Raj SK (25BCE2866)
### Deliverable 7: "Updating the project in GitHub repository (Link need to be shared during submission)"

Do this once, before you submit. It takes ~5 minutes. **Only you can do this step** because it needs your own GitHub account login.

## Step 1 — Create the repository
1. Sign in to [github.com](https://github.com) (create a free account if needed — use a professional username, this starts your digital profile).
2. Click **New repository**.
3. Repository name (suggested): `probabilistic-sms-spam-detection`
4. Description (paste this): `BAMAT207 project: SMS spam vs ham classification using prior/conditional probability, Bayes' theorem, expectation-variance and Multinomial Naive Bayes (UCI SMS Spam Collection).`
5. Choose **Public**, and do **NOT** tick "Add a README" / ".gitignore" / licence (this folder already contains them — ticking would cause a conflict).
6. Click **Create repository**. Copy the repository URL it shows, e.g. `https://github.com/YOUR-USERNAME/probabilistic-sms-spam-detection.git`
"https://github.com/SKNOMERCY/probabilistic-sms-spam-detection"

## Step 2A — Upload with Git (recommended)
Open a terminal **inside this project folder** (`BAMAT207_KavinRajSK_25BCE2866`) and run:

```bash
git init
git add .
git commit -m "BAMAT207: Probabilistic SMS spam detection (Kavin Raj SK, 25BCE2866)"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/probabilistic-sms-spam-detection.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. GitHub will ask you to sign in (use a Personal Access Token if it asks for a password: GitHub → Settings → Developer settings → Personal access tokens).

## Step 2B — No Git? Upload via the website instead
1. Open your new empty repository on github.com.
2. Click **uploading an existing file**.
3. Drag in **everything inside this folder** (report PDF/DOCX, notebook, `.py`, `README.md`, `requirements.txt`, and the `data/`, `figures/`, `model/`, `outputs/` folders).
4. Commit message: `BAMAT207: Probabilistic SMS spam detection (Kavin Raj SK, 25BCE2866)` → **Commit changes**.

## Step 3 — Verify (30 seconds)
- [ ] The repo page shows the README with the title, results table and charts.
- [ ] `figures/` images open; the notebook renders with outputs visible.
- [ ] The report PDF downloads from the repo.

## Step 4 — Share the link during submission
Submit this URL wherever the faculty asks for it:

```
https://github.com/YOUR-USERNAME/probabilistic-sms-spam-detection
```

(Write your real link here once created: ________________________________ )

> Tip: pin this repository on your GitHub profile — the activity sheet explicitly wants you to start building a digital profile, and this becomes your first public ML/statistics project.
