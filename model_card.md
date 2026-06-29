# Model Card: Mood Machine

This model card documents the Mood Machine project, which includes **two**
versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` (scikit-learn)

I built and compared **both** versions on the same labeled dataset.

## 1. Model Overview

**Model type:**
Both models. The rule based model is the primary one I engineered; the ML
model (bag-of-words + logistic regression) was used as a comparison baseline.

**Intended purpose:**
Classify short, social-media-style text messages into one of four moods:
`positive`, `negative`, `neutral`, or `mixed`. It is a learning exercise about
how data and modeling choices shape behavior — **not** a production system.

**How it works (brief):**
- *Rule based:* lowercases and tokenizes the text (keeping emojis and `:)`/`:(`
  as their own tokens), looks each token up in a positive/negative word list,
  adds/subtracts points, and maps the final score to a label.
- *ML:* turns each post into a bag-of-words count vector (`CountVectorizer`)
  and fits a `LogisticRegression` classifier on the labeled posts. It learns
  word→label associations automatically instead of using hand-written rules.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` / `TRUE_LABELS` contain **15 posts** total: the 6 starter posts
plus **9 new posts** I added in Part 1. The new posts deliberately include
slang (`lowkey`, `fire`, `slaps`), emojis (🔥, 💀, 🙂), sarcasm
(`"Oh great, another Monday meeting..."`), and mixed feelings
(`"Exhausted from finals but so grateful it's finally over"`).

**Labeling process:**
I labeled by the writer's **intended** mood, not the literal words. For
sarcasm that means the label is the opposite of the surface keyword — e.g.
`"Wow I just LOVE getting stuck in traffic for two hours"` is labeled
`negative` even though it contains "love". For mixed posts I used `mixed` when
two genuinely opposing feelings were present.

Posts that were hard to label / that a friend might disagree on:
- `"I'm fine 🙂"` — labeled `neutral`, but the smiley over "fine" can read as
  passive-aggressive or quietly negative.
- `"Oh great, another Monday meeting that could've been an email"` — labeled
  `negative` (sarcasm), but the literal word "great" pulls toward positive.
- `"Honestly today was just okay, nothing special"` — labeled `neutral`, but
  "just okay / nothing special" has a faintly disappointed tone.

**Important characteristics of the dataset:**
- Contains slang and emojis.
- Includes sarcasm (at least 2 clear cases).
- Several posts express mixed feelings.
- Posts are very short (a few words to one sentence).

**Possible issues with the dataset:**
- **Tiny** (15 examples) — far too small to train a reliable ML model.
- **Class imbalance** — only ~2 examples for some labels, which biases the ML
  model heavily.
- It only covers casual English/internet slang; no other dialects or languages.

## 3. How the Rule Based Model Works

**Your scoring rules:**
- Each positive word adds points; each negative word subtracts points.
- **Word weighting:** strong words (`love`, `hate`, `terrible`, `awful`,
  `amazing`, ...) count for **2** instead of 1. (`WORD_WEIGHTS`)
- **Negation handling:** a negation word (`not`, `never`, `n't`, ...) flips the
  sign of the **next** sentiment word, so `"not happy"` scores −1 and
  `"not bad"` scores +1. A non-sentiment word in between cancels the negation.
- **Emoji / emoticon signals:** `:)`, `:(`, and emojis like 🔥, 💀, 😭 are kept
  as tokens and scored. `fire🔥` is split into both `fire` and `🔥`.
- **Repeated-letter normalization:** `"soooo"` → `"soo"` so stretched spellings
  still match.
- **Threshold mapping (`predict_label`):**
  `score > 0 → positive`, `score < 0 → negative`. For `score == 0` I split:
  if there was both a positive **and** a negative hit it's `mixed`
  (e.g. "exhausted but grateful"); if there were no opposing signals it's
  `neutral` (e.g. "this is fine").

**Strengths of this approach:**
- Predictable and fully explainable — `explain()` shows exactly which words
  drove the score.
- Handles clear positives/negatives and simple negation well.
- Generalizes to **unseen** plain sentences better than the ML model does
  (see §5), because it doesn't depend on having seen the exact words before.

**Weaknesses of this approach:**
- **Sarcasm:** reads keywords literally, so `"I love getting stuck in traffic"`
  scores +2 → `positive` when a human means `negative`.
- **Unknown slang:** any word not in the list contributes 0. Before Part 3,
  `"This party is sick"` scored 0 → `neutral`.
- **No real context:** word order beyond immediate negation is ignored.

## 4. How the ML Model Works

**Features used:** Bag of words using `CountVectorizer` (raw word counts).

**Training data:** Trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`
(the same 15 rows the rule based model is evaluated on).

**Training behavior:** Accuracy on the training set is **1.00**. But this is
*training* accuracy — the model is graded on the exact rows it memorized, so a
perfect score is expected and misleading. Adding or relabeling even a few posts
visibly changes its predictions, because its entire "worldview" is those 15 rows.

**Strengths and weaknesses:**
- *Strength:* learns word→mood associations automatically and even gets the
  **training** sarcasm rows right (it memorized that "great" + "Monday meeting"
  → negative), which the rule based model cannot.
- *Weakness:* with only 15 imbalanced examples it **overfits** and fails to
  generalize — on new sentences it misclassifies obvious positives (see §5).

## 5. Evaluation

**How I evaluated:**
- Rule based: `python main.py` reports per-post predictions + accuracy on the
  15 labeled posts.
- ML: `python ml_experiments.py` reports training accuracy on the same posts.
- I also wrote `compare_models.py` to test both on **held-out** sentences
  (sentences not in the training set) — the honest test.

Headline numbers:
- **Rule based accuracy on `SAMPLE_POSTS`: 0.87** (13/15) after the Part 3 fix
  (it was 0.80 before adding the missing slang).
- **ML training accuracy: 1.00** (15/15) — but this overstates real ability.

**Examples of correct predictions (rule based):**
- `"Today was a terrible day"` → `negative` (score −2; "terrible" weighted).
- `"I am not happy about this"` → `negative` (negation flips "happy" to −1).
- `"Exhausted from finals but so grateful it's finally over"` → `mixed`
  (grateful +1 vs exhausted −1 → score 0 with opposing hits → mixed).

**Examples of incorrect predictions:**
- *Rule based, sarcasm:* `"Oh great, another Monday meeting that could've been
  an email"` → predicted `positive`, true `negative`. The single word "great"
  scored +1 and there was no negative keyword to balance it.
- *Rule based, sarcasm:* `"Wow I just LOVE getting stuck in traffic for two
  hours"` → predicted `positive` (score +2), true `negative`.

**How the two models' failures differ (held-out sentences):**

| sentence (not in training set)        | human    | rule based | ML        |
|---------------------------------------|----------|------------|-----------|
| `I absolutely love this`              | positive | positive   | **negative** |
| `I am so proud and grateful`          | positive | positive   | **negative** |
| `Oh great, my flight got cancelled`   | negative | **positive** | negative |
| `The concert was straight fire`       | positive | positive   | positive  |
| `I love getting stuck in traffic`     | negative | **positive** | negative |

- The **rule based** model fails on **sarcasm** (reads "great"/"love"
  literally).
- The **ML** model fails on **generalization** — it calls plain positives like
  "I absolutely love this" *negative*, because with 15 tiny, imbalanced
  examples it learned spurious associations. Its perfect training score did not
  transfer to new data.

## 6. Limitations

- **Sarcasm is the rule based model's core failure.** Traced end to end:

  ```
  "I absolutely love getting stuck in traffic"
  tokens : ['i','absolutely','love','getting','stuck','in','traffic']
  signal : "love" matched POSITIVE_WORDS, weight +2; no negative token
  score  : +2
  label  : positive          <-- WRONG (a human reads this as negative)
  why    : the only sentiment token is "love", used ironically. Irony lives in
           context/tone, not in token counts, so no word-list edit can fix it.
  ```

  The same trap fires on `"Oh great, another Monday meeting..."` (the word
  "great" scores +1 → `positive`, true `negative`). A keyword scorer cannot
  tell that a positive word is being used ironically — this is an inherent
  limit of the approach, not a missing word.
- **Unknown slang is silently ignored** by the rule based model — it only knows
  the words I hard-coded. `"This party is sick"` was `neutral` until I added
  "sick" in Part 3, and `"sick"` is itself ambiguous (can be negative).
- **The dataset is tiny and imbalanced (15 posts)**, so the ML model overfits
  and does not generalize.
- **No handling of longer text, context across a sentence, or tone** beyond
  immediate negation.
- The ML "1.00" accuracy is **training accuracy only** — there is no held-out
  test set in the provided harness, so the headline number flatters the model.

## 7. Ethical Considerations

- **Distress could be misread.** `"I'm fine 🙂"` is classified `neutral`, but in
  real life that exact phrasing can mask sadness or a cry for help. A mood tool
  that takes "fine" at face value could miss someone who needs support.
- **Dialect / community bias.** The vocabulary is casual internet English. Slang
  carries opposite meanings across communities ("sick", "wicked", 💀), and the
  model only knows the senses I chose. It would misread groups whose language
  isn't represented in my word lists or training posts.
- **Privacy.** Mood detection over personal messages is sensitive; inferring
  emotional state from someone's texts without consent is intrusive even when
  the model is "just" a word counter.
- **Overconfidence from numbers.** A "100% accurate" ML model invites misplaced
  trust; this card exists so the limitations travel with the model.

## 8. Ideas for Improvement

- Add **much more** labeled data and balance the classes.
- Add a **held-out test set** instead of reporting training accuracy.
- Use **TF-IDF** instead of raw counts, and add n-grams so phrases like
  "not bad" are learnable features for the ML model.
- Expand emoji/slang preprocessing and maintain the word lists more carefully.
- For sarcasm, move beyond keywords entirely (e.g. a small neural/transformer
  model that can use context) — rule based scoring fundamentally can't solve it.
- Keep `explain()`-style transparency in whatever model replaces this one.

---

### How to reproduce these results

```bash
pip install -r requirements.txt

# Rule based: per-post predictions, explanations, and accuracy (0.87)
python main.py          # press Enter to skip the interactive loop

# ML model: training accuracy (1.00)
python ml_experiments.py

# Part 3 stress test (breaker sentences)
python stress_test.py

# Part 4 honest comparison on held-out sentences
python compare_models.py
```

> On Windows, set `PYTHONUTF8=1` first so the emoji print correctly:
> `set PYTHONUTF8=1` (cmd) or `$env:PYTHONUTF8=1` (PowerShell).
