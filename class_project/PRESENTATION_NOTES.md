# Presentation outline and rehearsal guide

This is a preparation outline. Fill result slides from actual run outputs.

## 0-5 minutes: problem and methods

Say: "My application researches stocks. This study asks whether a model understands
financial sentiment and whether that information helps an existing predictor."
Explain positive/neutral/negative with one sentence each. Explain VADER as a rule-based
baseline, DistilBERT as a pretrained transformer we fine-tune, and LightGBM as trees
that combine numerical features. A sentiment score is not a stock-return forecast.

## 5-10 minutes: implementation

Show a simple flow: labeled sentences -> tokenizer -> DistilBERT -> sentiment.
Then: historical headlines -> sentiment feature -> LightGBM alongside existing factors.
Explain practice (train), checkpoint selection (development), final exam (test).
Show the 70/15/15 split and the actual training configuration. Point to reproducible
commands, frozen split IDs, model checkpoint and manifest.

## 10-15 minutes: results and lessons

Use the actual comparison table and confusion matrix. Explain accuracy versus macro-F1.
Show two mistakes and discuss ambiguity/contrast. Explain FinBERT's prior PhraseBank
exposure. Show the with/without-sentiment Brier comparison only if it completed.
If news data failed coverage checks, explain that limitation and do not invent results.

## 15-25 minutes: interactive demo

1. Enter a clear positive sentence, then a negative and a neutral one.
2. Compare VADER, our trained model and the reference.
3. Ask classmates to suggest a contrasting headline.
4. Show a saved error example and invite an explanation.
5. Open the financial results tab and explain one historical example.
6. Use labelled recorded screenshots if the runtime/network is unavailable.

## 25-30 minutes: Q&A preparation

**Why DistilBERT?** It is a small pretrained transformer with an established
classification workflow, practical for the available Colab GPU. We test, not assume,
that it improves over the simple baseline.

**What is fine-tuning?** Updating pretrained model weights on our labelled task.

**Why LightGBM?** The existing financial inputs are numerical tables, which boosted
decision trees handle well. We reuse it to isolate the effect of adding sentiment.

**Does better F1 mean better investment returns?** No. Sentiment recognition and
forecasting are separate; public information may already be reflected in prices.

**How is future information excluded?** NLP test labels do not select checkpoints.
Historical text is delayed and filtered before each feature date; financial training
labels must end before evaluation begins. The archive's timestamps remain a limitation.

**What exactly did you build?** Explain your original application contributions and
the class-specific training, baseline comparison, evaluation and demo. Attribute
pretrained models, published workflows, datasets and AI assistance accurately.

**Why no RNN?** It would add another model to explain without answering the main
question more directly. The course permits transformer text classification.

Rehearse using your own words. Team speaking roles can be distributed across the
required sections; check the lecturer's ten-minutes-per-member expectation.
