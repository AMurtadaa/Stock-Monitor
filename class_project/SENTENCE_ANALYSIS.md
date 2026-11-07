# Sentence analysis and transparent overall sentiment

## What changed

The demo now has a **Sentence analysis + weights** tab. It uses the same trained
seed-42 model. No new training, benchmark changes, financial feature changes or
trading behavior are involved. The original three-model comparison is preserved.

The goal is to reveal opposing information rather than hide it inside one label.
This is a research extension, not a demonstrated improvement in model accuracy.

## Run it in your existing Colab notebook

1. Stop the currently running demo cell.
2. Rerun section 1 to update the source and mount the existing Drive artifacts.
3. Run section 2. On a CPU session use:
   `app.setup("/content/stock-class-env-v2", require_gpu=False)`.
4. Run section 10 to launch the updated demo. No training or news rescoring is needed.
5. Open **Sentence analysis + weights** and paste a passage.

## Understand the calculations

For each displayed sentence (and optional separate headline):

`sentence score = P(positive) - P(negative)`

`effective weight = base weight * review multiplier`

`normalized weight = effective weight / sum(all effective weights)`

`contribution = normalized weight * sentence score`

`overall score = sum(all contributions)`

The overall score stays between -1 and +1, but is not a probability. Positive and
negative evidence can cancel. We report opposing sentence scores separately, so
near-zero is not automatically described as uniformly neutral text.

The displayed net direction uses fixed cutoffs above +0.15, below -0.15, or
near-zero in between. These are interpretation rules, not calibrated boundaries
or a newly trained class. Opposing evidence means at least one included unit has
a score above +0.15 and another below -0.15. A tiny-weight opposing unit can trigger
this flag: inspect the positive/negative weight shares as well.

## Which information gets more weight?

**Equal weights:** all base weights are 1. This is the baseline and default.

**Target relevance (experimental):** enter explicit aliases, for example
`Nasdaq, QQQ, semiconductors`. A sentence containing a complete alias gets base
weight 2; the remaining context keeps weight 1. This is a deliberately simple
heuristic, not learned financial significance or a claim that 2 is optimal.
The model does not automatically know QQQ means Nasdaq exposure. Mention matching
can miss relevant context and can overweight irrelevant mentions.

**Manual review:** the second table accepts multipliers from 0 to 5. A multiplier
of 0 excludes a unit, 1 preserves its weight and 2 doubles it. Explain your reason
for a change. Do not choose weights just to produce the label you want. Editing
weights changes the summary, not model probabilities. Headline status, sentence
position, sentiment strength and confidence do not independently increase weight.

An illustrative example (not measured model output): a +0.70 sentence about Nasdaq
and a -0.70 sentence about Dow average to zero under equal weights. Giving the
Nasdaq sentence twice the weight produces `(2*0.70 - 0.70)/3 = +0.2333`, with
opposing evidence still visible. This answers a target-focused question, not
which sentence is objectively most important to all investors.

## Read the evidence table

Each row shows the actual text, predicted class, score, base weight, review
multiplier, normalized share and contribution. The JSON also contains class
probabilities and the literal matched aliases. An exact repeated headline/body
unit is counted once. Paraphrased duplicates are not detected.

Splitting uses an offline punctuation heuristic with common abbreviation handling,
not an entity parser. It can mis-segment unusual abbreviations. A contrast within
one sentence ("profits rose but demand weakened") remains inside that sentence.

Each model input is limited to 128 tokens, including special tokens. A long unit
is flagged and a warning explains that its score does not cover all its text.
The original comparison now also reports input token counts/truncation. Sentence
decomposition often covers more of a paragraph, but does not guarantee full coverage.

## What this extension cannot claim

It does not identify event severity, credibility, causal impact, historical versus
future events, or expected stock returns. Article recency/source metadata are not
available in this text demo and are not invented. Multiple articles would need
separate identifiers, timestamps, duplicate handling and another evaluated rule.

To learn importance rather than hand-code it, we would need a defined target and
time horizon, human-labeled target-specific passages/evidence, and a separate
training/development/test design. More model confidence is not more importance.

## How we will evaluate it honestly

The original PhraseBank test metrics remain unchanged. Local logic tests and the
synthetic Gradio/model integration checks do not establish improved NLP accuracy.

Before claiming an improvement, collect a separate set of about 30–50 short
financial passages. Include single-direction, mixed, neutral, multi-entity,
historical/current and negation examples; use genuinely new examples, not altered
PhraseBank test sentences. For every passage, record:

- Original text, source/date and the target being judged.
- A human overall direction: positive, negative or neutral/near-zero.
- Whether opposing evidence is present, and the supporting text.
- An annotator explanation; a second independent reader where feasible.

Freeze labels and the weighting rule before running the comparison. Compare raw
paragraph output, equal sentence weights and fixed target weights on the same
examples. Report errors/disagreements, including segmentation and truncation.
Keep ambiguous labels visible rather than treating the model as the answer key.
Any changes developed from these errors need new held-out examples for evaluation.
Manual after-the-fact weights are a review demonstration, not a fair competitor.
No human-annotation accuracy results have been generated for this extension yet.

## Files

- `class_project/sentence_analysis.py`: splitting, matching, weighting and audit output.
- `class_project/demo.py`: Gradio inputs, editable weights and evidence tables.
- `class_project/tests/test_sentence_analysis.py`: synthetic logic and UI-callback checks.
- `class_project/tests/test_integration.py`: actual tiny-model integration.

An established research direction for the larger entity-specific task is
[SemEval-2017 Task 5](https://aclanthology.org/S17-2089/). Our simple literal
alias-matching extension is not equivalent to solving that task.
