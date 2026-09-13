"""
Hardcoded ASL-gloss -> English rules for the current small vocabulary.

ASL grammar differs from English (topic-comment structure, no "to be" verb,
combined pronouns like "minemy" for I/me/my), so a straight word-for-word
join doesn't read as English. This fills in the grammatical gaps we can
anticipate for this vocabulary with simple pattern matching.

Not a general solution - every rule here is specific to the current 11 words.
Once the vocabulary grows past what a few hardcoded patterns can cover, this
should be replaced with an LLM-based translation step instead.
"""

PRONOUNS = {"minemy": "I"}
PEOPLE = {"mom": "mom", "dad": "dad"}
ADJECTIVES = {"happy": "happy", "sad": "sad"}
STANDALONE = {
    "hello": "Hello.",
    "please": "Please.",
    "thankyou": "Thank you.",
    "yes": "Yes.",
    "no": "No.",
    "later": "See you later.",
}


def gloss_to_english(gloss_words):
    """Translate a short sequence of recognized ASL sign glosses into a
    plain English sentence, using hardcoded patterns for this vocabulary."""
    words = list(gloss_words)
    if not words:
        return ""

    # PERSON MINEMY ADJECTIVE, e.g. "mom minemy happy" -> "Mom, I am happy."
    if (
        len(words) == 3
        and words[0] in PEOPLE
        and words[1] in PRONOUNS
        and words[2] in ADJECTIVES
    ):
        return f"{PEOPLE[words[0]].capitalize()}, {PRONOUNS[words[1]]} am {ADJECTIVES[words[2]]}."

    # MINEMY ADJECTIVE, e.g. "minemy happy" -> "I am happy."
    if len(words) == 2 and words[0] in PRONOUNS and words[1] in ADJECTIVES:
        return f"{PRONOUNS[words[0]]} am {ADJECTIVES[words[1]]}."

    # MINEMY PERSON, e.g. "minemy mom" -> "My mom."
    if len(words) == 2 and words[0] in PRONOUNS and words[1] in PEOPLE:
        return f"My {PEOPLE[words[1]]}."

    # Single standalone word
    if len(words) == 1 and words[0] in STANDALONE:
        return STANDALONE[words[0]]

    # No pattern matched - fall back to a plain capitalized word list
    return " ".join(w.capitalize() for w in words) + "."
