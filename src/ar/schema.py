from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Condition = Literal[
    "base_bf16",
    "aligned_bf16",
    "base_quant",
    "aligned_quant",
    "base_quant_plus_adapter",
]


class TabooRecord(BaseModel):
    """One model response in the Taboo battery.

    The two sides of the taboo behaviour are logged separately and never combined
    into a single score, because they can dissociate and the two failure modes
    carry opposite implications:

      constraint lost, capability kept -> the model still knows the word and now
        says it. The alarming case, and the analogue of a safety fine-tune
        retaining its knowledge while losing its refusals.
      capability lost, constraint kept -> the model no longer hints at anything
        and trivially never says the word. Benign, and hard to distinguish from
        the base model.

    A single accuracy number would make these indistinguishable. This follows the
    same principle as the tool_attempted / tool_call_wellformed / tool_used split
    in the Phase 2 schema.
    """

    model_config = ConfigDict(frozen=True)

    # --- identity ---
    seed: int
    adapter: str
    secret_word: str
    base_model: str
    precision: str
    condition: Condition
    prompt_id: str
    prompt_text: str

    # --- raw output, never discarded ---
    response_text: str
    completion_tokens: int
    wall_time_s: float

    # --- CONSTRAINT side: did it avoid saying the word? ---
    said_word: bool
    """Secret word appears verbatim (case-insensitive, word-boundary) in the response."""
    said_word_count: int
    """How many times. A model that blurts once differs from one that repeats it."""
    first_word_position: int | None
    """Character index of the first occurrence, or None. Early leakage differs from
    a trailing slip."""

    # --- CAPABILITY side: does it still know the word? ---
    p_word_reveal: float
    """P(secret word) at a forced-reveal continuation. Measures retained knowledge
    independently of whether the model volunteers it."""
    logit_gap_reveal: float
    """logit(secret word) - max logit over other candidates at the same position.
    Positive means the word is the argmax. Scale-free complement to p_word_reveal."""
    word_rank_reveal: int
    """Rank of the secret word in the reveal-position distribution. 1 is argmax."""

    # --- P7 dose-response covariate, measured in-run at BF16 ---
    p_word_base: float
    """P(secret word) at the same position under the BASE model, i.e. the prior the
    suppression has to fight. Registered covariate for P7: constraint failure
    should scale with this."""

    # --- degenerate-output guards, so a broken generation is not read as behaviour ---
    is_empty: bool
    is_degenerate_repeat: bool
    """Same token or short cycle dominating the completion."""


class RetentionRecord(BaseModel):
    """Weight-space retention for one (adapter, layer, module, config, regime)."""

    model_config = ConfigDict(frozen=True)

    adapter: str
    base_model: str
    rank: int
    alpha: float
    use_rslora: bool
    effective_scaling: float
    layer: int
    module: str
    precision: str
    regime: Literal["fixed_scale", "adaptive_scale"]

    cosine: float
    relative_error: float
    retention_ratio: float
    projection_coefficient: float
    code_flip_rate: float
    value_change_rate: float
    subthreshold_fraction: float
    predicted_flip_rate: float
    mean_abs_delta_over_step: float
