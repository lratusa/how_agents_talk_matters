"""Frozen prompt templates, transcribed verbatim from preregistration §6.

Free-answer benchmarks replace `<letter>` with `<value>` and omit Options
(§6.1); the same letter->value substitution is applied to the review (§6.2)
and synthesizer (§6.3) templates, and to the constructed debate prompt.

The only non-verbatim prompt is the C7 debate-update prompt: prereg §6 freezes
only 6.1-6.3, so the debate prompt below is constructed in the same style
(marked DEVIATION in the README/report).
"""

LETTERS = "ABCDEFGHIJ"


def _answer_kind(mcq):
    return "<letter>" if mcq else "<value>"


def format_options(options):
    """Options labeled A..J (prereg supports up to 10 options)."""
    return "\n".join("%s. %s" % (LETTERS[i], opt) for i, opt in enumerate(options))


def solve_prompt(question, options):
    """§6.1 verbatim; free-answer variant drops the Options block."""
    mcq = options is not None
    prompt = (
        "Solve the following problem. Think step by step, then give a concise "
        "justification (at most 120 words) of the key reasoning. End your "
        "response with a single line exactly of the form:\n\n"
        "FINAL ANSWER: %s\n\n"
        "Problem:\n%s" % (_answer_kind(mcq), question)
    )
    if mcq:
        prompt += "\n\nOptions:\n%s" % format_options(options)
    return prompt


def review_prompt(question, own_answer, own_justification,
                  peer_answer, peer_justification, mcq):
    """§6.2 verbatim. For self-review (C9) the peer solution IS the agent's
    own response; the prompt is unchanged (no index leakage)."""
    return (
        "You are reviewing another independent solution to the same problem.\n\n"
        "Do not assume that your own answer is correct.\n"
        "Do not assume that the other answer is wrong.\n"
        "Check the peer solution independently.\n\n"
        "Identify:\n"
        "1. claims or steps you believe are correct;\n"
        "2. specific factual, mathematical, or logical errors;\n"
        "3. assumptions that require verification;\n"
        "4. whether the peer's conclusion should be retained or changed;\n"
        "5. your proposed corrected conclusion.\n\n"
        "Judge the reasoning, not writing style or confidence.\n\n"
        "Problem:\n%s\n\n"
        "Your own solution:\n"
        "Answer: %s\n"
        "Justification: %s\n\n"
        "Peer solution:\n"
        "Answer: %s\n"
        "Justification: %s\n\n"
        "End your review with a single line exactly of the form:\n"
        "REVIEW VERDICT: PEER CORRECT or PEER INCORRECT\n"
        "And then a line:\n"
        "YOUR UPDATED ANSWER: %s"
        % (question, own_answer, own_justification,
           peer_answer, peer_justification, _answer_kind(mcq))
    )


def synth_prompt(question, materials, with_reviews, mcq):
    """§6.3 verbatim; the optional clause is included iff with_reviews."""
    reviews_clause = ", and pairwise peer reviews of them," if with_reviews else ""
    return (
        "You are given several independent candidate solutions to the same "
        "problem%s with anonymized randomized labels. Produce the single best "
        "final answer.\n\n"
        "Rules:\n"
        "- Do not count repeated claims as independent evidence; the same "
        "reasoning may appear multiple times.\n"
        "- Evaluate argument quality, not rhetorical confidence or frequency.\n"
        "- Use the criticisms to locate specific errors.\n"
        "- Recover a correct minority solution if its reasoning is sound.\n"
        "- Ignore style and confidence.\n\n"
        "Problem:\n%s\n\n"
        "%s\n\n"
        "End with a single line exactly of the form:\n"
        "FINAL ANSWER: %s" % (reviews_clause, question, materials,
                              _answer_kind(mcq))
    )


def debate_prompt(question, materials, mcq):
    """C7 debate-update prompt (CONSTRUCTED — not frozen in prereg §6).
    Each agent sees the problem plus all N anonymized initial answers and
    produces an updated answer ending 'FINAL ANSWER: <x>'."""
    return (
        "You are given several independent candidate solutions to the same "
        "problem with anonymized randomized labels. Verify each candidate's "
        "reasoning independently, then give your own updated answer.\n\n"
        "Rules:\n"
        "- Do not assume any candidate is correct, including the one matching "
        "your own prior answer.\n"
        "- Check each candidate's reasoning step by step.\n"
        "- Ignore style and confidence.\n\n"
        "Problem:\n%s\n\n"
        "%s\n\n"
        "End with a single line exactly of the form:\n"
        "FINAL ANSWER: %s" % (question, materials, _answer_kind(mcq))
    )


def candidate_block(label, answer, justification):
    return "Candidate %s:\nAnswer: %s\nJustification: %s" % (
        label, answer, justification)


def review_block(reviewer_label, reviewee_label, review_text):
    return "Review of %s by %s:\n%s" % (reviewee_label, reviewer_label, review_text)
