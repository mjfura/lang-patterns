"""What every experiment asked, which models it trained, and what it found — the Streamlit page's
account of the work, in the order it was done. Each entry points at its README, which holds the
full numbers; `models` are glob patterns under `runs/` for the models an experiment trained.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Experiment:
    key: str
    phase: str
    title: str
    question: str
    design: str
    finding: str
    predictions: str = ""
    models: tuple[str, ...] = ()
    readme: str = ""
    corpus: dict[str, str] = field(default_factory=dict)  # run-name pattern -> corpus description


OVERVIEW = """
**Goal of the paper.** Indicators, computed on a corpus alone and before training, of whether a
language model can generalize the knowledge domain the corpus is about. The frame: a domain is a
bounded set of syntactic and semantic patterns; syntactic patterns are the base, and a corpus is
syntactically sufficient when each pattern of its domain is shown with enough different fillers
that the model learns the pattern rather than a word ("cut" with tomato, potato, onion… not only
with potato).

**How the work evolved.**

1. **The graph curves** (red, blue, orange, green): a transition graph over the model's BPE tokens,
   and curves of how much of it stays connected as more repetition is demanded. Contamination tests
   showed red and blue *rise* with duplicates, shuffled text and code: they measure how travelled
   a transition is (the trunk), not whether it forks (the branches). Only green partly separated
   noise.
2. **Held-out information** (bits per byte of n-gram models on unseen text): stable across merge
   budgets and correct on contamination, but it is an n-gram baseline, and it later turned out to
   rank corpora in *reverse* of generalization.
3. **Syntactic sufficiency** — the curves' idea made complete: induce coarse grammatical classes,
   read each pattern as a head followed by a class, and measure how many different fillers of the
   class it shows (coverage; later diversity, corrected for the head's traffic). Tested first on a
   synthetic recipe grammar with a trained model, then on real text, with gaps planted on purpose.
4. **Where it stops**: on a topic-narrow real corpus the model fails by topic, document-wide, not at
   syntactic slots. The indicator's claim is bounded to syntactic sufficiency; topic and semantic
   coverage is a separate dimension.

Every experiment below had its predictions written before it ran; the ones that failed are kept as
failed.
"""


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        key="curves",
        phase="1 · The graph curves",
        title="Merge sweep, subsampling, contamination, clean domains, null model",
        question="Do the red/blue/orange/green curves score a corpus's learnable structure?",
        design=(
            "The curves read on FineWeb at many BPE merge budgets and corpus sizes; FineWeb with "
            "10-50% of added control text, exact duplicates, Python code or word-shuffled text "
            "(3 seeds); eight clean sources against their own word-shuffled versions."
        ),
        finding=(
            "The readings depend on the merge budget and the corpus size. Red and blue rise with "
            "every contamination, noise included: they measure repetition, not patterns. Green is "
            "the only curve that falls with shuffled text and code. Comparisons across sources at "
            "one merge budget were withdrawn (different budgets give different tokens per word)."
        ),
        readme="experiments/contamination/README.md",
    ),
    Experiment(
        key="information",
        phase="2 · Held-out information",
        title="Predictability and order in bits per byte",
        question="Does a held-out n-gram measure behave as a learnability indicator should?",
        design=(
            "Smoothed n-gram models (orders 0-3) fitted on half the documents and scored on the "
            "other half, per byte; order = the word-shuffled null's score minus the corpus's."
        ),
        finding=(
            "Stable within 0.04 bits per byte across merge budgets; more predictable with new "
            "documents, unchanged with duplicates, less with shuffled text. But it is an n-gram "
            "baseline, and on the recipe grammar it ranked the corpora in reverse of what the "
            "model generalized."
        ),
        readme="experiments/information/README.md",
    ),
    Experiment(
        key="recipes",
        phase="3 · Syntactic sufficiency, synthetic",
        title="Recipe grammar: does coverage predict whether the model ties a verb to its nouns?",
        question=(
            "If each verb is shown with fewer ingredients, does the model treat unseen "
            "ingredients as invalid after that verb — and does the corpus's coverage predict it?"
        ),
        design=(
            "A recipe grammar where each verb is seen with 1, 2, 4, 8, 16, 20, 24 or all 28 of its "
            "ingredients (2000 documents). The same model trained on each; per verb, the "
            "generalization ratio: the share of probability after '<verb> the' going to "
            "ingredients never seen with that verb, over their fair share (1 = pattern learned, "
            "0 = memorized). Three seeds, two model sizes."
        ),
        finding=(
            "Coverage predicts the ratio verb by verb: Spearman 0.94-0.975 on every seed and "
            "size. The rise is sharp: the small model generalizes near coverage 0.8 (24 of 28), "
            "the larger one needs more. The held-out loss ranks the variants in reverse."
        ),
        predictions="First test 3 of 4 held; replication 4 of 4 held.",
        models=("syntactic_sufficiency/small/*", "syntactic_sufficiency/medium/*"),
        readme="experiments/syntactic_sufficiency/README.md",
        corpus={"*": "synthetic recipe grammar; the run name gives ingredients per verb and seed"},
    ),
    Experiment(
        key="dynamics",
        phase="3 · Syntactic sufficiency, synthetic",
        title="Recipe grammar: corpus size and training dynamics",
        question="Does the model's generalization depend on corpus size and on how long it trains?",
        design=(
            "The small model on 500, 2000 and 8000 documents of 7 recipe variants, 3 seeds, read "
            "at steps 250, 500, 1000, 2000, 3000 and at its best checkpoint."
        ),
        finding=(
            "Every model first treats all ingredients alike, then learns each verb's restriction. "
            "Coverage predicts where training ends up (Spearman 0.955-0.974 on 2000 and 8000 "
            "documents); sufficiency is relative to the model and its training budget. On 500 "
            "documents the class induction fell to 2 classes, a fault found and fixed here."
        ),
        predictions="1 of 4 held; the failures are recorded with their causes.",
        models=("syntactic_sufficiency/dynamics/*/*",),
        readme="experiments/syntactic_sufficiency/README.md",
        corpus={
            "*": "synthetic recipe grammar; documents, ingredients per verb and seed in the name"
        },
    ),
    Experiment(
        key="natural",
        phase="4 · Real text",
        title="First reading on eight real sources",
        question=(
            "On real text, do the classes look grammatical and are the flagged patterns readable?"
        ),
        design=(
            "Classes and patterns read on arXiv, PubMed, Wikipedia, C4, FineWeb, FineWeb-Edu, "
            "case law and MultiNLI."
        ),
        finding=(
            "No class groups by topic, and several are clean categories (determiners, singular and "
            "plural nouns, participles), but word pieces and markup fill some. The low-coverage "
            "list is debris plus fixed constructions (due to, able to, didn't), which raw coverage "
            "cannot tell from missing data."
        ),
        predictions="1 partly met, 1 failed.",
        readme="experiments/natural_sufficiency/README.md",
    ),
    Experiment(
        key="reference",
        phase="4 · Real text",
        title="Fixed construction or missing data: a reference corpus",
        question="Can a larger corpus of the same domain tell a construction from a gap?",
        design=(
            "Patterns over word-like tokens only; diversity (fillers against what chance would "
            "give at the head's traffic); a slot low in the corpus but diverse in a reference is "
            "'under-shown', low in both is 'fixed'."
        ),
        finding=(
            "due, able, pursuant, such are fixed in every source; restricted recipe verbs are "
            "under-shown. The 0.5 cut is too permissive against the model (it generalizes near "
            "0.8)."
        ),
        predictions="4 of 4 held.",
        readme="experiments/reference_diagnosis/README.md",
    ),
    Experiment(
        key="planted",
        phase="4 · Real text",
        title="Gaps planted in real text, and how strong they must be",
        question=(
            "If words are tied to few fillers on purpose, does the indicator find them, and only "
            "them?"
        ),
        design=(
            "FineWeb, Wikipedia, arXiv, C4 in 8 slices; five heads per source restricted to their "
            "2, 4, 8, 16 or 32 most frequent fillers in one slice; a flag needs a drop beyond the "
            "pattern's own noise across the unplanted slices."
        ),
        finding=(
            "Found 19, 20, 19, 17, 12 of 20 heads at 2, 4, 8, 16, 32 fillers, with no false alarm "
            "among some 1,200 other patterns. Removing sentences instead of replacing words lost "
            "the heads' traffic and hid the gap."
        ),
        predictions="Mixed: detection held where the head kept its traffic; failures recorded.",
        readme="experiments/planted_gap/README.md",
    ),
    Experiment(
        key="planted_model",
        phase="5 · Closing the loop",
        title="A model trained on a gapped corpus fails where the indicator flags",
        question=(
            "Does a model trained on the gapped text fail on unseen text exactly at the flagged "
            "heads?"
        ),
        design=(
            "FineWeb slices 0-6 as they are (control) or with five heads kept to 2 or 16 fillers; "
            "slice 7, never planted, evaluates. Per head: the log probability of the true next "
            "word, gapped model minus control."
        ),
        finding=(
            "The indicator flags all five heads. After them the true word is 2.2-4.6 nats less "
            "likely with 2 fillers kept (9 to 100 times) and 1.1-3.6 with 16, on every head and "
            "seed; five comparison heads stay near the control."
        ),
        predictions="3 of 5 held; one comparison reading and the global-loss prediction failed.",
        models=("planted_model/*",),
        readme="experiments/planted_model/README.md",
        corpus={
            "control_*": "FineWeb slices 0-6, unmodified",
            "k2_*": "FineWeb slices 0-6, five heads kept to 2 fillers",
            "k16_*": "FineWeb slices 0-6, five heads kept to 16 fillers",
        },
    ),
    Experiment(
        key="natural_gap",
        phase="5 · Closing the loop",
        title="A natural gap: topic-narrow corpora against a random one",
        question=(
            "Does the indicator predict where a model trained on a topic-narrow real corpus does "
            "worse than one trained on a random corpus of the same size?"
        ),
        design=(
            "FineWeb clustered by topic; 3.5 MB of health+sports+education, and of conversational "
            "blogs, against 3.5 MB drawn at random; 400 held-out documents evaluate. Diversity and "
            "reference coverage (which of the domain's fillers a slot shows) read on each."
        ),
        finding=(
            "Neither measure predicts where the narrow models lose. Their loss is by topic: better "
            "than the random model on documents of their own topics, worse on all others "
            "(+0.11 to +0.15 bits per byte). A topic-narrow corpus fails a model semantically, "
            "document-wide; the indicator's claim is bounded to syntactic sufficiency."
        ),
        predictions="Diversity: 1 of 3 held. Reference coverage: 1 of 3 held.",
        models=("natural_gap/*",),
        readme="experiments/natural_gap/README.md",
        corpus={
            "narrow_*": "FineWeb, health + sports + education clusters (3.5 MB)",
            "narrow2_*": "FineWeb, conversational-blog cluster (3.5 MB)",
            "random_*": "FineWeb, random documents (3.5 MB)",
        },
    ),
    Experiment(
        key="subset_selection",
        phase="5 · Closing the loop",
        title="Choosing a training subset, with the full corpus as reference",
        question=(
            "Does a subset's diagnosis against its full corpus say, before training, which cut "
            "keeps the full corpus's generalization and where the others lose it?"
        ),
        design=(
            "FineWeb pool (14.4 MB) as full corpus and reference; random cuts of 50, 25 and 10%, "
            "and two 25% perplexity-filtered cuts (CCNet-style against Wikipedia; the pool's own "
            "most formulaic text). corpus-diagnose on each; the small model on each and on the "
            "full pool, seeds 0 and 1; 400 held-out documents evaluate."
        ),
        finding=(
            "No false alarms on random cuts (0.15 to 0.21% of rows, chance level) and more flags "
            "on the formulaic cut, but the flags are mild (diversity 0.26 to 0.50) and the models "
            "do not lose more at those heads. A flag that is significant is not yet large: "
            "choosing data needs an effect size on top of the margin."
        ),
        predictions="2 of 5 held; 1 split (one filter held, one failed); 2 failed.",
        models=("subset_selection/*",),
        readme="experiments/subset_selection/README.md",
        corpus={
            "full_*": "FineWeb pool, 4,600 documents (14.4 MB)",
            "r50_*": "FineWeb pool, random 50%",
            "r25_*": "FineWeb pool, random 25%",
            "r10_*": "FineWeb pool, random 10%",
            "wiki25_*": "FineWeb pool, 25% lowest perplexity under a Wikipedia bigram",
            "self25_*": "FineWeb pool, 25% lowest perplexity under its own bigram",
        },
    ),
    Experiment(
        key="dose_response",
        phase="5 · Closing the loop",
        title="Dose and response: how far must diversity fall before a model pays?",
        question=(
            "As a head's diversity is lowered step by step, from 0.7 to 0.1, how much does a "
            "model trained on the corpus lose on that head's real text?"
        ),
        design=(
            "FineWeb sample of planted_model; 14 heads, each restricted with a probability "
            "calibrated so its diversity hits 0.1 to 0.7 (two heads per level, two assignments "
            "so each head is read at two doses); 3 comparison heads untouched. The small model, "
            "seeds 0 and 1; the planted_model controls reused."
        ),
        finding=(
            "Monotone and strong (Spearman 0.76 to 0.92; 3 nats lost at 0.1), and it starts "
            "early: half a nat already at 0.5 to 0.6, crossing 0.3 nats at 0.64. The 0.5 cut is "
            "too permissive for this restriction. The natural cuts' mild flags cost nothing, so "
            "the loss follows how far the filler distribution moved, not how many fillers show."
        ),
        predictions="3 of 6 held (monotone, strong end, comparison heads).",
        models=("dose_response/*",),
        readme="experiments/dose_response/README.md",
        corpus={
            "A_*": "FineWeb slices 0 to 6, 14 heads at assignment A doses",
            "B_*": "FineWeb slices 0 to 6, the same heads at assignment B doses",
        },
    ),
    Experiment(
        key="divergence",
        phase="5 · Closing the loop",
        title="A divergence as the effect size",
        question=(
            "Does KL(reference fillers || corpus fillers) per head, which has the loss's units, "
            "predict where a model pays better than diversity does?"
        ),
        design=(
            "No training: dose_response and subset_selection re-read with D, the KL from the "
            "reference's fillers of a head to the corpus's, smoothed with Witten-Bell backoff; "
            "then D in excess of 8 random cuts of F at each subset's size, and one score per cut."
        ),
        finding=(
            "Drop is about -0.68 times D (R² 0.77) on plantings, where D ranks no better than "
            "diversity. On natural cuts D ranks the per-head loss (Spearman -0.39 to -0.45) where "
            "flags did not, but that was sample size alone. Against 8 random cuts of the same "
            "size, the excess separates the filters (z 3.9 and 15) from random cuts (|z| < 0.6) "
            "and points, narrowly, at the heads where a filter's model loses more (Spearman "
            "-0.21 to -0.28). It does not order whole models: wiki25 costs more by topic."
        ),
        predictions="2 of 5 held; against random cuts 3 of 4 held (not the model order).",
        readme="experiments/divergence/README.md",
    ),
    Experiment(
        key="cut_choice",
        phase="5 · Closing the loop",
        title="Choosing among cuts of one size: does the score order the models?",
        question=(
            "Among ten cuts of FineWeb at 25% of its bytes, does the cut with the lower "
            "divergence score train the better model?"
        ),
        design=(
            "Three cuts of subset_selection plus seven new ones (random, the opposite ends of "
            "both perplexity filters, shortest and longest documents, half self25). Each scored "
            "against 8 random cuts of the same size; the small model, seeds 0 and 1."
        ),
        finding=(
            "The per-head reading replicates on every filter (Spearman -0.23 to -0.38), and along "
            "one filter's dose the score orders the models. Across kinds of filter it orders them "
            "only partly (0.55 / 0.45): document count, a proxy for topics, matters about as much "
            "(the shortest-documents cut trains the best model, the 108 longest the worst). "
            "Two dose ladders, mixing a filter with random documents: with documents held, the "
            "models follow the score (Spearman 0.90); when the dose also adds documents, they "
            "win at low doses and the slot loss from a score of about 0.05."
        ),
        predictions=(
            "3 of 4 held (not the order over all ten cuts); ladders 1 of 3 held, 1 split "
            "(self held, wikihigh failed)."
        ),
        models=("cut_choice/*",),
        readme="experiments/cut_choice/README.md",
        corpus={
            "r25b_*": "FineWeb pool, random 25% (seed 203)",
            "r25c_*": "FineWeb pool, random 25% (seed 204)",
            "selfhigh25_*": "FineWeb pool, 25% highest perplexity under its own bigram",
            "wikihigh25_*": "FineWeb pool, 25% highest perplexity under a Wikipedia bigram",
            "short25_*": "FineWeb pool, 25% shortest documents",
            "long25_*": "FineWeb pool, 25% longest documents",
            "mix25_*": "FineWeb pool, half self25's order, half random",
            "self_m*": "FineWeb pool, self25's order mixed with random at the dose in the name",
            "wikihigh_m*": "FineWeb pool, wikihigh25's order mixed with random at the dose",
        },
    ),
    Experiment(
        key="syntactic_eval",
        phase="5 · Closing the loop",
        title="A syntactic evaluation on natural text",
        question=(
            "Does the indicator predict whether a model prefers, after a head, an unseen word of "
            "the right class (spaCy tags) over one of another class, rather than the true word?"
        ),
        design=(
            "No training: the 6 planting models and 36 cut models re-read with 20 frequency-"
            "matched pairs x 20 real contexts per head; classes from spaCy, never from the "
            "indicator."
        ),
        finding=(
            "All six predictions failed. Class knowledge breaks only when a slot collapses (the "
            "strongest plantings); among realistic cuts it follows size and document count, not "
            "the score. The indicator predicts which filler a cut costs (the drop), and class-"
            "level failure only for collapsed slots."
        ),
        predictions="0 of 6 held.",
        readme="experiments/syntactic_eval/README.md",
    ),
    Experiment(
        key="class_patterns",
        phase="5 · Closing the loop",
        title="Head patterns and class patterns: at which level does a model learn syntax?",
        question=(
            "If she and they lose the verb after them but he, we and I keep it, does the model "
            "still know a verb goes after she? And if every pronoun loses it?"
        ),
        design=(
            "Sentences removed from r50 at matched size: every pronoun/modal + verb sentence "
            "(class_cut), only those of she, they, can, will (head_cut), or random ones "
            "(control). Small model, seeds 0 and 1; G from syntactic_eval."
        ),
        finding=(
            "For pronouns: she and they keep their syntax when only they lose the slot (-0.04 "
            "G) and lose it with the class (-0.18; every pronoun about -0.2). Class-level D "
            "predicts this and head-level D does not. Replicated with valid tests on modals "
            "(-0.13) and possessives (-0.30): removing the class pattern removes syntax for "
            "every member; removing it from two heads costs them 25-57% as much."
        ),
        predictions=(
            "2 of 4 held; replication on modals and possessives 6 of 7; the tool's class rows "
            "found 1 of 3 removed class patterns with no false flag."
        ),
        models=("class_patterns/*",),
        readme="experiments/class_patterns/README.md",
        corpus={
            "class_cut_*": "r50 without any pronoun/modal + verb sentence, 4.67 MB",
            "head_cut_*": "r50 without she/they/can/will + verb sentences, 4.67 MB",
            "control_*": "r50 without random sentences, 4.67 MB",
            "poss_class_cut_*": "r50 without any possessive + noun sentence, 5.59 MB",
            "poss_head_cut_*": "r50 without his/their + noun sentences, 5.59 MB",
            "poss_control_*": "r50 without random sentences, 5.59 MB",
        },
    ),
    Experiment(
        key="model_sweep",
        phase="5 · Closing the loop",
        title="Corpus or model? Attributing a syntactic failure",
        question=(
            "If the tool flags a class pattern, can any model recover its syntax? If it flags "
            "nothing, does some model reach the reference's syntax?"
        ),
        design=(
            "F, r10 and the size-matched unflagged controls and flagged class cuts of "
            "class_patterns, each trained small, medium (4 layers, 256 wide) and long (9000 "
            "steps); G on pronouns, modals and possessives."
        ),
        finding=(
            "Flagged class patterns stay 0.13-0.27 below F under every configuration, and larger "
            "or longer-trained models do worse on them. Unflagged controls reach F within 0.02. "
            "A flag means the corpus is at fault; no flag means a failure is the model's."
        ),
        predictions="3 of 3 held.",
        models=("model_sweep/*",),
        readme="experiments/model_sweep/README.md",
        corpus={
            "full_*": "FineWeb pool F (14.4 MB)",
            "r10_*": "FineWeb pool, random 10%",
            "control_*": "r50 without random sentences, 4.67 MB",
            "class_cut_*": "r50 without any pronoun/modal + verb sentence, 4.67 MB",
            "poss_control_*": "r50 without random sentences, 5.59 MB",
            "poss_class_cut_*": "r50 without any possessive + noun sentence, 5.59 MB",
        },
    ),
    Experiment(
        key="baselines",
        phase="5 · Closing the loop",
        title="Do existing corpus metrics see what the indicator sees?",
        question=(
            "On the same class cuts, do perplexity, unigram KL, MATTR, distinct-2 or an "
            "embedding Vendi score detect what the indicator flags?"
        ),
        design=(
            "Six one-number metrics on 4 class cuts, 4 head cuts and 8 random controls, all "
            "sentence-level edits of r50; no training."
        ),
        finding=(
            "As planned, none flags the class cuts, but that test mixed control sizes. Size-"
            "matched, unigram KL and distinct-2 register the two large cuts (10-18k sentences) "
            "yet stay in noise on the small ones (2-4k) that the indicator flags, and none "
            "says which pattern changed."
        ),
        predictions="2 of 2 held as written (lenient; see README).",
        readme="experiments/baselines/README.md",
    ),
    Experiment(
        key="blimp",
        phase="5 · Closing the loop",
        title="The models on BLiMP",
        question=(
            "How do the models score on the field's syntax benchmark, and does it see class cuts?"
        ),
        design="The 30 existing models scored on BLiMP's 67 paradigms by total log probability.",
        finding=(
            "F's models read 0.56-0.58 (chance 0.5): beyond local slot classes, these small "
            "models know little syntax. BLiMP moves at most 0.017 between control and class "
            "cut where G moves 0.11-0.34: one global score does not localize."
        ),
        predictions="1 of 2 held.",
        readme="experiments/blimp/README.md",
    ),
)
