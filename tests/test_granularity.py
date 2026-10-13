import random
import string

from corpus_generalization.synthetic.recipes import LEXICON, recipe_documents, verb_pairing
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer
from corpus_generalization.tokenizer.granularity import (
    learn_to_granularity,
    tokens_per_word_by_budget,
)


def _recipes(count: int) -> list[str]:
    return recipe_documents(LEXICON, verb_pairing(LEXICON, None, 2, 0), count, 0)


def test_the_curve_is_the_encoding_at_every_budget() -> None:
    documents = [*_recipes(30), "Two <|im_start|> markers <|im_end|> encode to their own ids."]
    learned = ByteLevelBPETokenizer.from_document_sequence(documents, recognize_markers=True)
    curve = tokens_per_word_by_budget(documents, learned.merges)
    words = sum(len(document.split()) for document in documents)
    assert len(curve) == len(learned.merges) + 1
    for budget in (0, 1, 50, 151, len(learned.merges)):
        cut = ByteLevelBPETokenizer.from_merges(learned.merges[:budget])
        tokens = sum(len(cut.encode(document, recognize_markers=True)) for document in documents)
        markers = 2  # reserved ids, which no piece of `_training_pieces` counts
        assert curve[budget] == (tokens - markers) / words


def test_learning_to_a_granularity_stops_at_the_fewest_merges_that_reach_it() -> None:
    documents = _recipes(30)
    full = ByteLevelBPETokenizer.from_document_sequence(documents, recognize_markers=True)
    curve = tokens_per_word_by_budget(documents, full.merges)
    tokenizer, reached = learn_to_granularity(documents, 1.5)
    budget = len(tokenizer.merges)
    assert tokenizer.merges == full.merges[:budget]
    assert reached == curve[budget] <= 1.5 < curve[budget - 1]


def test_learning_resumes_past_the_first_round_and_keeps_every_merge_when_they_run_out() -> None:
    generator = random.Random(0)
    words = ["".join(generator.choices(string.ascii_lowercase, k=7)) for _ in range(600)]
    documents = [" ".join(generator.choices(words, k=40)) for _ in range(200)]
    full = ByteLevelBPETokenizer.from_document_sequence(documents, recognize_markers=True)
    # Below what the corpus can reach: learning resumes past its first 1,024 merges, then stops
    # on its own.
    tokenizer, reached = learn_to_granularity(documents, 0.5)
    assert len(full.merges) > 1024
    assert tokenizer.merges == full.merges
    assert reached == tokens_per_word_by_budget(documents, full.merges)[-1] > 0.5
