"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    # Added in Part 1/2: common praise words and informal positives.
    "proud",
    "hopeful",
    "grateful",
    "wonderful",
    "fantastic",
    "perfect",
    "lit",          # slang: exciting / great
    "vibes",        # slang: usually positive ("good vibes")
    # Part 3 targeted fix: positive slang the model was missing (scored 0).
    "fire",         # slang: "this is fire" = great (the WORD, not just 🔥)
    "sick",         # slang: "that's sick" = awesome (ambiguous! see limitations)
    "slaps",        # slang: "the soundtrack slaps" = really good
    "goated",       # slang: greatest of all time
    ":)",           # emoji handled as its own token in preprocess()
    "😂",
    "🥰",
    "🔥",           # slang/emoji: "fire" = great
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    # Added in Part 1/2: stronger/weaker negatives and informal negatives.
    "exhausted",
    "annoyed",
    "anxious",
    "disappointed",
    "worst",
    "horrible",
    "miserable",
    "ugh",          # slang interjection signaling frustration
    ":(",
    "😭",
    "💀",           # ambiguous slang; here treated as negative ("dead/done")
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    # ---- New posts added in Part 1 ----
    "Lowkey stressed but kind of proud of myself",
    "This new album is straight fire 🔥",
    "Oh great, another Monday meeting that could've been an email",
    "I'm fine 🙂",
    "Wow I just LOVE getting stuck in traffic for two hours",
    "ugh my wifi died right before the deadline 💀",
    "Honestly today was just okay, nothing special",
    "Exhausted from finals but so grateful it's finally over",
    "not gonna lie this movie was kinda bad but the soundtrack slaps",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    # ---- New labels added in Part 1 (must stay aligned with SAMPLE_POSTS) ----
    "mixed",     # "Lowkey stressed but kind of proud of myself" -> stress + pride
    "positive",  # "This new album is straight fire 🔥" -> slang "fire" = great
    "negative",  # "Oh great, another Monday meeting..." -> SARCASM (edge case)
    "neutral",   # "I'm fine 🙂" -> deliberately ambiguous; could be masked sadness
    "negative",  # "Wow I just LOVE getting stuck in traffic..." -> SARCASM (edge case)
    "negative",  # "ugh my wifi died right before the deadline 💀" -> frustration
    "neutral",   # "Honestly today was just okay, nothing special" -> flat tone
    "mixed",     # "Exhausted from finals but so grateful it's over" -> tired + grateful
    "mixed",     # "this movie was kinda bad but the soundtrack slaps" -> con + pro
]
#
# EDGE CASES we (and a friend) might disagree on:
#   - "I'm fine 🙂"            : labeled neutral, but the smiley over "fine" can
#                                read as passive-aggressive / quietly negative.
#   - "Oh great, ... email"    : labeled negative because it's sarcasm, but the
#                                literal word "great" makes a keyword model say
#                                positive. Classic rule-based failure.
#   - "Wow I just LOVE ..."    : same sarcasm trap as above ("love" is positive
#                                literally, negative in intent).
#
# These are exactly the rows we expect the rule-based model to get wrong in
# Part 3 / Part 4.

# Safety check: the program (and ML training) will crash later if these two
# lists ever drift out of alignment, so assert it right where the data lives.
assert len(SAMPLE_POSTS) == len(TRUE_LABELS), (
    f"SAMPLE_POSTS ({len(SAMPLE_POSTS)}) and TRUE_LABELS "
    f"({len(TRUE_LABELS)}) must have the same length."
)
