from typing import Protocol, runtime_checkable

BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"


# The specials live with the Protocol rather than inside one tokenizer, because their ids are
# what the two implementations have to agree on: a dataset writes `<bos>` and `<pad>` into its
# tensors by id, so a tokenizer that numbered them differently would silently reinterpret every
# sequence the other one framed.
#
# The two chat markers are reserved here, beside the other three, so that no text can ever spell
# them. `encode` answers with byte ids and merge ids and nothing else, so a user message holding
# the characters `<|im_start|>` comes back as those characters; the marker is an id only
# `eva_llm.data.chat` emits, and only between turns. Reserving them is what makes a turn boundary
# a fact about the sequence rather than a substring anyone can write.
SPECIAL_TOKENS = (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, IM_START_TOKEN, IM_END_TOKEN)
BOS_ID, EOS_ID, PAD_ID, IM_START_ID, IM_END_ID = range(len(SPECIAL_TOKENS))


@runtime_checkable
class Tokenizer(Protocol):
    """Contract shared by every tokenizer, whatever its segmentation strategy.

    `encode` is text to the ids that spell it and nothing more. Framing a sequence with
    `<bos>`/`<eos>`, marking its turns with `<|im_start|>`/`<|im_end|>` and squaring a batch off
    with `<pad>` are decisions about a training sequence, not about segmentation, so they belong
    to the dataset — which is also the only place that can make them, since padding needs the
    whole batch to know how long the longest sequence is. `decode` sees through that framing,
    because what comes back from a model carries it.
    """

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...
