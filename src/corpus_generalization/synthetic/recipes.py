"""A recipe grammar with a closed lexicon, in which how many fillers each pattern is seen with is a
parameter and some combinations are never generated, so a model's generalization to them can be
tested with minimal pairs.

Four sentence rules: an action (`cut the ripe tomatoes .`), an action with a manner (`stir the
onion gently .`), an action with a place (`fry the potatoes in the pan .`), and a state with
agreement (`the onions are soft .`). `verb_pairing` draws the reserved combinations from the seed
before anything else, so the diverse and restricted variants of one seed hold out the same ones and
are tested on the same minimal pairs.
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Ingredient:
    singular: str
    plural: str


@dataclass(frozen=True)
class Lexicon:
    verbs: tuple[str, ...]
    ingredients: tuple[Ingredient, ...]
    adjectives: tuple[str, ...]
    adverbs: tuple[str, ...]
    places: tuple[tuple[str, str], ...]  # (preposition, tool)


def _ingredients(*forms: str) -> tuple[Ingredient, ...]:
    return tuple(Ingredient(*form.split("/")) for form in forms)


# fmt: off
LEXICON = Lexicon(
    verbs=(
        "cut", "chop", "slice", "peel", "wash", "dice", "grate",
        "mash", "roast", "fry", "boil", "add", "mix", "stir",
    ),
    ingredients=_ingredients(
        "tomato/tomatoes", "potato/potatoes", "onion/onions", "carrot/carrots",
        "pepper/peppers", "apple/apples", "lemon/lemons", "egg/eggs",
        "mushroom/mushrooms", "cucumber/cucumbers", "eggplant/eggplants", "bean/beans",
        "leek/leeks", "radish/radishes", "beet/beets", "pear/pears",
        "peach/peaches", "plum/plums", "lime/limes", "banana/bananas",
        "mango/mangoes", "avocado/avocados", "cabbage/cabbages", "shallot/shallots",
        "chili/chilies", "olive/olives", "cherry/cherries", "fig/figs",
        "zucchini/zucchinis", "turnip/turnips",
    ),
    adjectives=(
        "fresh", "ripe", "small", "large", "soft", "firm",
        "sweet", "sour", "hot", "cold", "raw", "clean",
    ),
    adverbs=("slowly", "gently", "quickly", "carefully", "well", "finely"),
    places=(
        ("in", "pan"), ("in", "pot"), ("in", "bowl"), ("in", "oven"),
        ("with", "knife"), ("with", "spoon"), ("on", "board"), ("on", "tray"),
    ),
)
# fmt: on

# Rule weights: action, action + manner, action + place, state.
RULE_WEIGHTS = (0.35, 0.2, 0.2, 0.25)
ADJECTIVE_PROBABILITY = 0.3
PLURAL_PROBABILITY = 0.5
SENTENCES_PER_DOCUMENT = (4, 10)


@dataclass(frozen=True)
class Pairing:
    """Which ingredients each verb is generated with (singular forms), and the combinations no
    document ever holds: `(verb, ingredient)` for actions, `(ingredient, adjective)` for states."""

    allowed: Mapping[str, tuple[str, ...]]
    reserved_actions: frozenset[tuple[str, str]]
    reserved_states: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class MinimalPair:
    kind: Literal["verb", "agreement"]
    good: str
    bad: str


def verb_pairing(
    lexicon: Lexicon, fillers_per_verb: int | None, reserved_per_verb: int, seed: int
) -> Pairing:
    """Reserve `reserved_per_verb` ingredients per verb and one adjective per ingredient, then give
    each verb `fillers_per_verb` of its remaining ingredients (all of them when `None`)."""
    nouns = [ingredient.singular for ingredient in lexicon.ingredients]
    available = len(nouns) - reserved_per_verb
    if fillers_per_verb is not None and not 1 <= fillers_per_verb <= available:
        raise ValueError(f"fillers_per_verb must be in 1..{available}, got {fillers_per_verb}")

    rng = random.Random(seed)
    reserved = {verb: set(rng.sample(nouns, reserved_per_verb)) for verb in lexicon.verbs}
    reserved_states = frozenset((noun, rng.choice(lexicon.adjectives)) for noun in nouns)

    # A separate stream, so the reserved draws do not depend on how many fillers are kept.
    fillers_rng = random.Random(f"{seed}:fillers")
    allowed: dict[str, tuple[str, ...]] = {}
    for verb in lexicon.verbs:
        remaining = [noun for noun in nouns if noun not in reserved[verb]]
        if fillers_per_verb is not None:
            remaining = fillers_rng.sample(remaining, fillers_per_verb)
        allowed[verb] = tuple(remaining)
    return Pairing(
        allowed=allowed,
        reserved_actions=frozenset((verb, noun) for verb in reserved for noun in reserved[verb]),
        reserved_states=reserved_states,
    )


def recipe_documents(lexicon: Lexicon, pairing: Pairing, count: int, seed: int) -> list[str]:
    """`count` documents of 4 to 10 sentences each."""
    rng = random.Random(seed)
    forms = {ingredient.singular: ingredient for ingredient in lexicon.ingredients}
    adjectives_of = {
        noun: [
            adjective
            for adjective in lexicon.adjectives
            if (noun, adjective) not in pairing.reserved_states
        ]
        for noun in forms
    }

    def noun_form(noun: str) -> tuple[str, bool]:
        plural = rng.random() < PLURAL_PROBABILITY
        return (forms[noun].plural if plural else noun), plural

    def action() -> list[str]:
        verb = rng.choice(lexicon.verbs)
        noun, _ = noun_form(rng.choice(pairing.allowed[verb]))
        adjective = [rng.choice(lexicon.adjectives)] if rng.random() < ADJECTIVE_PROBABILITY else []
        return [verb, "the", *adjective, noun]

    def sentence() -> list[str]:
        rule = rng.choices(range(4), weights=RULE_WEIGHTS)[0]
        if rule == 0:
            return action()
        if rule == 1:
            return [*action(), rng.choice(lexicon.adverbs)]
        if rule == 2:
            preposition, tool = rng.choice(lexicon.places)
            return [*action(), preposition, "the", tool]
        noun = rng.choice(list(forms))
        form, plural = noun_form(noun)
        return ["the", form, "are" if plural else "is", rng.choice(adjectives_of[noun])]

    return [
        " ".join(" ".join([*sentence(), "."]) for _ in range(rng.randint(*SENTENCES_PER_DOCUMENT)))
        for _ in range(count)
    ]


def minimal_pairs(lexicon: Lexicon, pairing: Pairing, seed: int) -> list[MinimalPair]:
    """One verb pair per reserved action (an adverb in the noun's slot) and two agreement pairs per
    reserved state (singular and plural, the other form of to be)."""
    rng = random.Random(seed)
    plurals = {ingredient.singular: ingredient.plural for ingredient in lexicon.ingredients}
    pairs = [
        MinimalPair("verb", f"{verb} the {noun} .", f"{verb} the {rng.choice(lexicon.adverbs)} .")
        for verb, noun in sorted(pairing.reserved_actions)
    ]
    for noun, adjective in sorted(pairing.reserved_states):
        plural = plurals[noun]
        pairs += [
            MinimalPair(
                "agreement", f"the {noun} is {adjective} .", f"the {noun} are {adjective} ."
            ),
            MinimalPair(
                "agreement", f"the {plural} are {adjective} .", f"the {plural} is {adjective} ."
            ),
        ]
    return pairs


FOREIGN_SUBJECTS = ("bank", "market", "fund", "company", "investor", "broker")
FOREIGN_VERBS = ("raises", "lowers", "sells", "buys", "reports", "reviews")
FOREIGN_OBJECTS = ("rate", "price", "stock", "profit", "share", "loan")
FOREIGN_TIMES = ("today", "again", "yearly")


def foreign_documents(count: int, seed: int) -> list[str]:
    """Documents from a small finance grammar sharing no content word with the recipe lexicon, for
    the outlier variant."""
    rng = random.Random(seed)

    def sentence() -> str:
        time = [rng.choice(FOREIGN_TIMES)] if rng.random() < 0.3 else []
        words = [
            "the",
            rng.choice(FOREIGN_SUBJECTS),
            rng.choice(FOREIGN_VERBS),
            "the",
            rng.choice(FOREIGN_OBJECTS),
            *time,
            ".",
        ]
        return " ".join(words)

    return [
        " ".join(sentence() for _ in range(rng.randint(*SENTENCES_PER_DOCUMENT)))
        for _ in range(count)
    ]
