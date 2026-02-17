# logic_mapper.py

import re


def classify_logic(question: str) -> str:
    """
    Decide which logical framework fits the question
    """

    q = question.lower()

    # Proof / math
    if any(w in q for w in ["prove", "theorem", "derive", "calculate", "show that"]):
        return "formal"

    # Probability / uncertainty
    if any(w in q for w in ["probability", "chance", "likely", "random", "bayes"]):
        return "probabilistic"

    # Paradox / contradiction
    if any(w in q for w in ["paradox", "contradiction", "liar", "false", "self reference"]):
        return "paraconsistent"

    # Philosophy / meaning
    if any(w in q for w in ["meaning", "life", "truth", "free will", "consciousness", "existence"]):
        return "philosophical"

    # Causal / science
    if any(w in q for w in ["why", "cause", "effect", "reason", "mechanism"]):
        return "causal"

    # Default
    return "classical"


def wrap_prompt(question: str):
    """
    Build structured prompt for the model
    """

    logic_type = classify_logic(question)


    if logic_type == "formal":
        system = (
            "[Logic-Type: Formal]\n"
            "Use mathematical notation and logical proof.\n"
        )

    elif logic_type == "probabilistic":
        system = (
            "[Logic-Type: Probabilistic]\n"
            "Use Bayesian reasoning and probability theory.\n"
        )

    elif logic_type == "paraconsistent":
        system = (
            "[Logic-Type: Paraconsistent]\n"
            "Allow contradictions without collapse.\n"
        )

    elif logic_type == "philosophical":
        system = (
            "[Logic-Type: Philosophical]\n"
            "Analyze conceptually and abstractly.\n"
        )

    elif logic_type == "causal":
        system = (
            "[Logic-Type: Causal]\n"
            "Explain using cause-effect models.\n"
        )

    else:
        system = (
            "[Logic-Type: Classical]\n"
            "Use standard logical reasoning.\n"
        )


    prompt = f"""
{system}

Interpret the question under this framework.

Question: {question}

Answer:
"""

    return prompt.strip(), logic_type
