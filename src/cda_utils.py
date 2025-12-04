import re

_GENDER_SWAP = {
    "he": "she",
    "she": "he",
    "him": "her",
    "her": "him",
    "his": "her",
    "man": "woman",
    "woman": "man",
    "men": "women",
    "women": "men",
    "boy": "girl",
    "girl": "boy",
    "boys": "girls",
    "girls": "boys",
    "male": "female",
    "female": "male",
    "father": "mother",
    "mother": "father",
    "son": "daughter",
    "daughter": "son",
    "brother": "sister",
    "sister": "brother",
    "husband": "wife",
    "wife": "husband",
}


def swap_gender_terms(text: str) -> str:
    """
    Perform simple Counterfactual Data Augmentation by swapping
    gendered words using _GENDER_SWAP. Case is preserved.
    """
    def repl(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in _GENDER_SWAP:
            swapped = _GENDER_SWAP[lower]
            if word[0].isupper():
                swapped = swapped.capitalize()
            return swapped
        return word

    # Replace whole words only
    return re.sub(r"\b\w+\b", repl, text)
