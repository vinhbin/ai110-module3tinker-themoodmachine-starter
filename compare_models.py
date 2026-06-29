"""
Part 4 helper: compare the rule based and ML models on HELD-OUT sentences
(sentences that are NOT in SAMPLE_POSTS), to expose the difference between
the ML model's perfect *training* accuracy and its real generalization.

Run with: set PYTHONUTF8=1 && python compare_models.py
"""

from ml_experiments import train_ml_model, predict_single_text
from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS

vec, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
rb = MoodAnalyzer()

# Sentences NOT in the training set -> this is the honest test.
held_out = [
    ("I absolutely love this", "positive"),
    ("This was the worst day ever", "negative"),
    ("Oh great, my flight got cancelled", "negative"),   # sarcasm, unseen
    ("The concert was straight fire", "positive"),       # known slang, new sentence
    ("I am so proud and grateful", "positive"),
    ("It is what it is", "neutral"),
    ("I love getting stuck in traffic", "negative"),     # sarcasm, unseen
]

header = f"{'sentence':45} {'human':9} {'rule':9} {'ML':9}"
print("=== Part 4: rule based vs ML on HELD-OUT sentences ===\n")
print(header)
print("-" * len(header))
for s, human in held_out:
    rb_pred = rb.predict_label(s)
    ml_pred = predict_single_text(s, vec, model)
    print(f"{s[:44]:45} {human:9} {rb_pred:9} {ml_pred:9}")
