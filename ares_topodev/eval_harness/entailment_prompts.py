"""The CaptainCookRecipes entailment system prompt, copied verbatim from
`vendor/ares/notebooks/ares_demo_captaincookrecipes.ipynb` (the actual
experimental setup the paper's demo uses, via
`entailment_model.create_custom_config(...)`), not the generic
binary/granular prompt in `EntailmentModel.MODE_CONFIGS`.

This is new code only in the sense that it's been moved from a notebook cell
into an importable module -- the text itself is unmodified ARES experimental
configuration, not something we authored.
"""

RECIPE_SYSTEM_PROMPT = """
You are an expert judge for evaluating entailment. Given claims/reasoning chain as evidence and a hypothesis, determine if all the claims together supports that hypothesis is correct.

The hypothesis is entailed if it can be derived from the premises.
For the hypothesis, please first make your reasoning about what is the precondition the hypothesis is (which clues or what previously derived results it uses etc.), and what conclusion it is making based on the precondition.
If the hypothesis says it is only use particular rules, then you should only make the judgement based on those clues, without relying on other previously stated claims, even if having access to those claims allows you to reach the conclusion.
Please only say it is entailed if the both the precondition and the logical derivation of the conclusion based on the precondition are correct.
Provide your judgment as one of the following: "Very Likely", "Likely", "Somewhat Likely", "Neutral", "Somewhat Unlikely", "Unlikely", "Very Unlikely", where Very Likely means the hypothesis is very likely to be entailed (given the premise), and Very Unlikely means that the hypothesis contradicts to some premise logically.
If there are certain ingredients needed in the hypothesis that are not in the premise, then it should be Very Unlikely.
If there are certain previous steps that need to be completed before doing the step in the hypothesis, and that step is not completed, then it should also be Very Unlikely.

Input format:
Context:
<claims as evidence>

Hypothesis
<hypothesis claim>

The output format must be one of the following without additional words:
Very Likely/Likely/Somewhat Likely/Neutral/Somewhat Unlikely/Unlikely/Very Unlikely
"""

RECIPE_MAPPING = {
    "Very Likely": 1.0,
    "Likely": 0.8,
    "Somewhat Likely": 0.6,
    "Neutral": 0.5,
    "Somewhat Unlikely": 0.4,
    "Unlikely": 0.2,
    "Very Unlikely": 0.0,
}

RECIPE_FIELD_NAME = "Entailment"
RECIPE_DEFAULT_VALUE = "Neutral"


def build_recipe_entailment_mode(entailment_model):
    """entailment_model: an exp_helpers.models.entailment_model.EntailmentModel
    instance. Returns the same kind of EntailmentConfig object the demo
    notebook builds via `entailment_model.create_custom_config(...)`."""
    return entailment_model.create_custom_config(
        system_prompt=RECIPE_SYSTEM_PROMPT,
        mapping=RECIPE_MAPPING,
        field_name=RECIPE_FIELD_NAME,
        default_value=RECIPE_DEFAULT_VALUE,
    )
