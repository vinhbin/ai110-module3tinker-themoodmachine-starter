# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


# Words that flip the meaning of the next sentiment word ("not happy").
NEGATION_WORDS = {"not", "no", "never", "n't", "dont", "cant", "wont"}

# A few words carry stronger feeling than a generic hit, so weight them.
# (value = points added/subtracted instead of the default 1)
WORD_WEIGHTS = {
    "love": 2,
    "amazing": 2,
    "awesome": 2,
    "perfect": 2,
    "fantastic": 2,
    "hate": 2,
    "terrible": 2,
    "awful": 2,
    "horrible": 2,
    "worst": 2,
    "miserable": 2,
}

# Emoji / punctuation "tokens" we want preprocess() to keep as standalone
# signals instead of stripping them away with the rest of the punctuation.
EMOTICONS = {":)", ":-)", ":(", ":-(", ":/", ":')"}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps (in order):
          1. Lowercase and strip outer whitespace.
          2. Split on whitespace into rough tokens.
          3. For each token, keep emoticons (":)", ":(") and emoji characters
             as their own signal tokens, but strip surrounding punctuation off
             of normal words ("traffic," -> "traffic", "love!!!" -> "love").
          4. Collapse runs of 3+ repeated letters ("soooo" -> "soo") so
             "loveee" still matches "love".

        Returns a flat list of tokens. We intentionally KEEP emojis instead of
        deleting them, because this dataset uses 🔥 / 💀 / 🙂 as mood signals.
        """
        cleaned = text.strip().lower()
        raw_tokens = cleaned.split()

        tokens: List[str] = []
        for raw in raw_tokens:
            # Keep known text emoticons exactly as-is.
            if raw in EMOTICONS:
                tokens.append(raw)
                continue

            # Pull any single-codepoint emoji out as their own tokens so a word
            # glued to an emoji ("fire🔥") still yields both "fire" and "🔥".
            for ch in raw:
                if self._is_emoji(ch):
                    tokens.append(ch)

            # Strip emoji + punctuation from the word, keeping letters/digits
            # and apostrophes (so "don't" survives as "don't").
            word = "".join(
                ch for ch in raw
                if (ch.isalnum() or ch == "'") and not self._is_emoji(ch)
            )
            # Normalize stretched spelling: 3+ repeats -> 2 ("soooo" -> "soo").
            word = re.sub(r"(.)\1{2,}", r"\1\1", word)
            if word:
                tokens.append(word)

        return tokens

    @staticmethod
    def _is_emoji(ch: str) -> bool:
        """True for emoji-range codepoints we care about (rough but enough)."""
        return ord(ch) >= 0x1F000 or ch in {"❤", "✅"}

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Core scoring routine shared by score_text(), predict_label(), and
        explain() so they can never disagree with each other.

        Returns (score, positive_hits, negative_hits).

        Enhancements implemented (Part 2):
          * Word weighting  -- strong words like "love"/"hate" count for 2,
            everything else counts for 1 (see WORD_WEIGHTS).
          * Negation        -- a negation word ("not", "never", "n't") flips
            the sign of the NEXT sentiment word, so "not happy" scores -1 and
            "not bad" scores +1.
          * Repeated words   -- each occurrence counts, so "love love love"
            scores higher than a single "love".
        """
        tokens = self.preprocess(text)

        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negate = False  # set True right after we see a negation word

        for token in tokens:
            if token in NEGATION_WORDS:
                negate = True
                continue

            weight = WORD_WEIGHTS.get(token, 1)

            if token in self.positive_words:
                if negate:
                    score -= weight
                    negative_hits.append(f"not+{token}")
                else:
                    score += weight
                    positive_hits.append(token)
            elif token in self.negative_words:
                if negate:
                    score += weight
                    positive_hits.append(f"not+{token}")
                else:
                    score -= weight
                    negative_hits.append(token)
            else:
                # A non-sentiment word in between cancels a dangling negation,
                # so "not really that happy" doesn't flip "happy".
                negate = False
                continue

            # Negation only applies to the single next sentiment word.
            negate = False

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score, negative words decrease it, and
        the enhancements described in _analyze() (weighting + negation) apply.
        """
        score, _positive_hits, _negative_hits = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        Our mapping (Part 2 threshold choices):
          - score > 0                     -> "positive"
          - score < 0                     -> "negative"
          - score == 0 AND both a positive
            and negative signal were seen -> "mixed"
          - score == 0 with no signals
            (or signals that cancel)       -> "neutral"

        The "mixed" vs "neutral" split matters: "exhausted but grateful" and
        "this is fine" can both score 0, but they are not the same mood. We
        use the presence of *opposing* hits to tell them apart.
        """
        score, positive_hits, negative_hits = self._analyze(text)

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"

        # score == 0 here: distinguish a real tug-of-war from a flat statement.
        if positive_hits and negative_hits:
            return "mixed"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        Uses the same _analyze() routine as score_text()/predict_label(), so
        the explanation always matches the actual decision (negation and word
        weighting included). Hits prefixed with "not+" were flipped by a
        preceding negation word.
        """
        score, positive_hits, negative_hits = self._analyze(text)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
