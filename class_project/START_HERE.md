# Start here: your financial sentiment class project

## What this project is

You are studying **financial sentiment classification** and one small financial
application. You are not presenting the whole Stock-Monitor application.

Your two questions are:

1. Can our fine-tuned language model correctly label financial text?
2. Does adding those sentiment scores improve an existing financial predictor?

VADER is a vocabulary-and-rules baseline. DistilBERT is the transformer we fine-tune.
LightGBM combines numerical features for the stock-prediction experiment. FinBERT is
a pretrained reference: it already trained on Financial PhraseBank, so its score on
that dataset is not an independent unseen-data result.

## Exactly what to do first

1. Open https://colab.research.google.com/ .
2. Choose **File > Open notebook > GitHub**.
3. Paste `https://github.com/AMurtadaa/Stock-Monitor` and select branch
   `class/financial-sentiment-study`.
4. Open `class_project/Financial_Sentiment_Study.ipynb`. Alternatively upload that
   notebook from your local `stock-monitor-class/class_project` folder.
5. Choose **Runtime > Change runtime type > GPU**.
6. Start at section 1 and run the cells in order. Read the explanation above each cell.

This notebook is DIFFERENT from the earlier SQuAD question-answering notebook.
There is no `nlp_colab_bundle.zip` upload for this version: it loads your class fork.

## What happens in each stage

### Setup

The notebook clones the class branch and creates an isolated Python environment.
If you authorize Drive mounting, outputs are saved under
`MyDrive/CSCI390_Financial_Sentiment/class_artifacts`.
Your ordinary notebook packages and deployed application are not modified.
Colab environments can lack pip even after virtual-environment creation. Setup
explicitly installs pip into the target using the notebook interpreter's
`pip --python` option before installing project dependencies.

Expected output: package versions, GPU name, and successful tests. Installation can
take several minutes. A new Colab session may need installation again; saved data and
completed models on Drive are reused.

### Dataset preparation

Financial PhraseBank contains human-labeled financial sentences. We use the
75%-agreement subset: 3,453 source lines. After removing duplicates/conflicts, our
verified preparation produced 3,448 unique examples: 2,413 training, 517 development,
and 518 test. The exact manifest is saved with every run.

Training is the practice material. Development chooses the best training checkpoint.
Test is the final exam. Do not change the method after looking at test errors and then
report the same test score as independent evidence.

### Smoke test

Two optimizer steps confirm that training, saving, reloading and prediction work.
Smoke evaluation uses development examples, not final test examples. The scores can
be low; do not submit them as final results.

### Full training

Train three seeds: 42, 17, 73. Each is three passes (epochs) through the training data.
The model starts from general-language DistilBERT, not our old QA checkpoint.
Progress bars show steps and estimated remaining time. The exact time depends on the
GPU; this dataset is much smaller than SQuAD. Use actual progress rather than a fixed
time estimate. Interrupted runs resume from their latest saved checkpoint.

The best checkpoint is selected using development macro-F1. The final test does not
select checkpoints or seeds. Seed 42 is fixed in advance for Experiment B.

### Sentiment evaluation

Inspect `evaluation/full/metrics.json`, the confusion matrix and training curves.
Accuracy is the fraction correct. Macro-F1 gives negative, neutral and positive equal
importance even though neutral examples are more common. The three-seed mean and
sample standard deviation describe run-to-run variation, not statistical certainty.

Open `review_15_examples.json`. Explain at least five mistakes in your own words.
Useful questions: Was the sentence ambiguous? Did it contain negation? Did the model
focus on the wrong clause? Does a neutral statement merely contain a positive word?

### Historical financial experiment

Sentiment labels alone do not establish financial prediction. This step needs a
separate history of headlines, prices and filing-dated fundamentals.

The notebook first checks for existing class inputs. A fresh Git clone intentionally
does not contain private databases. If you already have the FNSPID news CSV, supply
its path. Otherwise the default streams the public ~23 GB Nasdaq archive, which may
take a long time even though only our 12-stock subset is retained. Do this once.

Headline scoring is cached by checkpoint fingerprint and original text. It resumes
without rescoring completed text. We use a 48-hour availability delay, a seven-day
window and at most 20 unique headlines per stock/day. The delay reduces timestamp
risk; it does not prove that archive timestamps are correct.

SEC asks for a contact User-Agent. Enter your own contact email in Colab's prompt;
never send API secrets to chat. SEC access can be blocked from Colab; a clear error
means we should prepare the snapshot locally, not invent fundamentals.

The predictor asks whether a stock beats SPY over the next 12 months. We evaluate
monthly examples. Initial training is 2015-2018, development is 2020, refit includes
eligible 2015-2020 examples, and final evaluation is 2022-2023. Rows are purged by
actual outcome-end dates, and later prices are used only for labels.

Read `financial/coverage.json` before interpreting any model score. If there are too
few examples, missing stocks, one class, or less than 10% news coverage in a split,
the run stops. Share that report; do not change dates to seek a better result.

LightGBM is trained twice using the SAME rows and parameters: one variant has no
sentiment column; one adds sentiment. Brier score measures probability error (lower
is better). ROC-AUC measures ranking quality. Accuracy uses a fixed 0.5 threshold.
These are raw probabilities, without rank calibration.

This is a small descriptive prediction study. It is not a trading-return test.
The fixed surviving-company panel and overlapping annual labels limit the conclusion.

### Demo, export, and what to send me

Run writeup/export before the demo. Download the archive. The demo cell stays running
while its temporary public Gradio link is active; stopping it closes the website.

Send `submission/results_summary.json` for review, plus any error output. Keep the
model archive for the demo. Do not send your production database or credentials.
The generated report is a DRAFT, not a fabricated completed submission. We will write
the final interpretation and format the report after reviewing actual results.

## If something goes wrong

- Red traceback: stop at that cell, copy the last error and the cell name.
- No GPU: select a GPU runtime; do not run full training accidentally on CPU.
- Invalid `matplotlib_inline` backend: run `os.environ["MPLBACKEND"] = "Agg"`
  in the notebook kernel, then rerun setup. Class subprocesses save charts as PNGs
  and do not need Colab's inline plotting backend. The updated source sets this too.
- Missing model: complete training before evaluation; don't switch to a fallback.
- Missing news: finish the news step or supply an existing archive.
- Weak coverage: share coverage.json; don't treat all-zero sentiment as a result.
- Colab disconnect: reconnect, rerun setup, and reuse the same Drive folder.
- Completed-run message: the saved model is being reused; it is not training again.

## What exists versus what remains

Implemented: data preparation, strict model backend, seeded training, metric scoring,
plots, historical-news filtering/scoring, frozen finance snapshots, matched comparison,
demo, export, report draft generator, tests and this student workflow.

Results pending: full GPU runs, full sentiment scores and the real historical financial
comparison. The code cannot establish those results before the data and runs exist.
Final report and slides must use verified results and your reviewed explanations.

## Local commands (optional; Colab runs these for you)

From the isolated fork, install class dependencies in a separate environment and
ensure PyTorch is installed for your hardware. Run:

```text
python -m class_project check
python -m pytest class_project/tests -q
python -m class_project prepare
python -m class_project train --smoke
python -m class_project evaluate --smoke
python -m class_project train --seed 42
python -m class_project train --seed 17
python -m class_project train --seed 73
python -m class_project evaluate --finbert
python -m class_project filter-news --csv PATH_TO_FNSPID_CSV
python -m class_project score-news
python -m class_project snapshot --sec-user-agent "ClassResearch/1.0 (YOUR_EMAIL)"
python -m class_project compare
python -m class_project writeup
python -m class_project export
python -m class_project demo --finbert
```

## Sources and attribution

- Upstream Stock-Monitor: https://github.com/benjaminbob21/Stock-Monitor
- Hugging Face classification: https://huggingface.co/docs/transformers/tasks/sequence_classification
- Financial PhraseBank: https://huggingface.co/datasets/takala/financial_phrasebank
- DistilBERT: https://huggingface.co/distilbert/distilbert-base-uncased
- FinBERT: https://huggingface.co/ProsusAI/finbert
- FNSPID: https://huggingface.co/datasets/Zihan1004/FNSPID

Respect each dataset/model licence. Financial PhraseBank is CC BY-NC-SA 3.0; pretrained
FinBERT's card specifies CC BY-NC-SA 4.0. Existing application code retains its upstream
licensing. Record AI assistance and personal contributions according to course policy.
