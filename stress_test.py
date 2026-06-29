"""
Part 3: Stress test ("break the model").

Runs a set of deliberately tricky sentences through the rule based
MoodAnalyzer and prints, for each one:
  - the tokens the model saw
  - which words counted positive / negative
  - the final score and predicted label
  - what a human would actually say (expected)

Run with UTF-8 so the emoji print on Windows:
    set PYTHONUTF8=1 && python stress_test.py
"""

from mood_analyzer import MoodAnalyzer

# (sentence, what a human would label it) -- the "expected" is just for us to
# eyeball where the model disagrees with a person.
BREAKERS = [
    ("I love getting stuck in traffic", "negative"),    # sarcasm
    ("Oh great, my flight got cancelled", "negative"),  # sarcasm
    ("This party is sick", "positive"),                 # slang: sick = good
    ("That movie was wicked good", "positive"),         # slang: wicked = very
    ("The new update is fire", "positive"),             # slang: fire = great
    ("I'm fine", "negative"),                           # masked / flat
    ("I'm exhausted but proud of myself", "mixed"),     # genuinely mixed
    ("not bad at all honestly", "positive"),            # double negative-ish
    ("It is what it is", "neutral"),                    # idiom, no keywords
    ("great. just great.", "negative"),                 # sarcastic repetition
]


def main() -> None:
    analyzer = MoodAnalyzer()
    print("=== Part 3: Stress Test (breaker sentences) ===\n")
    for text, expected in BREAKERS:
        tokens = analyzer.preprocess(text)
        predicted = analyzer.predict_label(text)
        reason = analyzer.explain(text)
        flag = "OK" if predicted == expected else "MISS"
        print(f"[{flag}] \"{text}\"")
        print(f"       tokens   : {tokens}")
        print(f"       {reason}")
        print(f"       predicted: {predicted}   | a human would say: {expected}\n")


if __name__ == "__main__":
    main()
