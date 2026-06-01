"""Trader "expert" personas and the binary scenarios they judge.

The demonstration: three trader personas with *different dispositions*
(optimistic, neutral, pessimistic) each read the same set of one-line market
scenarios and make a binary call -- ``bullish`` or ``bearish``. Their
dispositions bias the ambiguous calls in different directions, which is exactly
the error structure ntqr needs: three noisy judges whose mistakes are not
identical.

Each scenario carries a ``truth`` (the defensible call given the stated facts)
so we can run a *supervised* ntqr evaluation as a check -- but the headline
result comes from the *unsupervised* algebraic evaluator, which recovers each
persona's accuracy from agreement structure alone, with no answer key.

Binary labels follow ntqr's convention of ``("a", "b")``:
    "bullish" -> "a"   (expect price up)
    "bearish" -> "b"   (expect price down)
"""

from __future__ import annotations

import hashlib
import json

LABELS = ("a", "b")
CALL_TO_LABEL = {"bullish": "a", "bearish": "b"}
LABEL_TO_CALL = {"a": "bullish", "b": "bearish"}

# Bump this when the actual prompt contract changes. The value is part of the
# LLM response-cache key, so changing prompts creates a new experiment record
# without deleting or refreshing the older stochastic cache.
PROMPT_VERSION = "v3_binary_schema"

CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "call": {
            "type": "string",
            "enum": ["bullish", "bearish"],
        }
    },
    "required": ["call"],
    "additionalProperties": False,
}

# Three personas -> a trio, matching ntqr's three-classifier R=2 evaluators.
PERSONAS = [
    {
        "key": "optimist",
        "name": "Optimist",
        "system": (
            "You classify next-quarter equity direction from the stated facts "
            "only. Use the Optimist decision policy: in genuinely mixed cases, "
            "give slightly more weight to upside catalysts and recovery paths. "
            "If the evidence is exactly balanced, choose bullish. "
            "Do not force optimism when explicit negative evidence dominates; "
            "such cases should be bearish. Return valid JSON only."
        ),
    },
    {
        "key": "neutral",
        "name": "Neutral",
        "system": (
            "You classify next-quarter equity direction from the stated facts "
            "only. Use the Neutral decision policy: weigh positive and "
            "negative evidence symmetrically, with no directional prior. Make "
            "the call the evidence best supports; if it is close, still choose "
            "the stronger stated side. Return valid JSON only."
        ),
    },
    {
        "key": "pessimist",
        "name": "Pessimist",
        "system": (
            "You classify next-quarter equity direction from the stated facts "
            "only. Use the Pessimist decision policy: in genuinely mixed "
            "cases, give slightly more weight to downside risks and execution "
            "failure. If the evidence is exactly balanced, choose bearish. "
            "Do not force pessimism when explicit positive evidence "
            "dominates; such cases should be bullish. Return valid JSON only."
        ),
    },
]

USER_TEMPLATE = (
    "Market scenario:\n{scenario}\n\n"
    "Task: classify the asset's expected direction over the next quarter.\n"
    "Use only the facts stated above; do not assume outside market data.\n"
    "Choose exactly one allowed value for call: bullish or bearish.\n"
    "No neutral, no mixed, no abstention, and no explanation are allowed. "
    "If evidence is mixed, choose the allowed side implied by your persona "
    "policy.\n"
    "Return exactly one JSON object and no other text, for example "
    '{{"call":"bullish"}} or {{"call":"bearish"}}.'
)

REPAIR_TEMPLATE = (
    "{user_prompt}\n\n"
    "Your previous response was invalid for this binary schema: {reason}.\n"
    "Previous response:\n{raw}\n\n"
    "Return exactly one JSON object with key call and value either bullish or "
    "bearish. No neutral, no mixed, no abstention, and no explanation."
)


def prompt_payload() -> dict:
    """Stable prompt payload included in summaries and cache keys."""
    return {
        "version": PROMPT_VERSION,
        "personas": [
            {
                "key": persona["key"],
                "name": persona["name"],
                "system": persona["system"],
            }
            for persona in PERSONAS
        ],
        "user_template": USER_TEMPLATE,
        "call_schema": CALL_SCHEMA,
        "labels": list(CALL_TO_LABEL),
    }


def prompt_hash() -> str:
    """Short stable hash of the exact prompt contract."""
    payload = json.dumps(
        prompt_payload(), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# Each scenario has a defensible directional call given only the stated facts.
# A few are intentionally ambiguous to create persona disagreement.
SCENARIOS = [
    {
        "id": "s01",
        "truth": "bullish",
        "text": "Revenue grew 42% YoY, beat estimates, and management raised full-year guidance.",
    },
    {
        "id": "s02",
        "truth": "bearish",
        "text": "The company missed earnings for the third straight quarter and cut its dividend.",
    },
    {
        "id": "s03",
        "truth": "bullish",
        "text": "A blockbuster drug just received FDA approval ahead of schedule with a broad label.",
    },
    {
        "id": "s04",
        "truth": "bearish",
        "text": "Regulators opened a fraud probe and the CFO abruptly resigned this morning.",
    },
    {
        "id": "s05",
        "truth": "bullish",
        "text": "Free cash flow doubled, the firm initiated a large buyback, and debt fell sharply.",
    },
    {
        "id": "s06",
        "truth": "bearish",
        "text": "A key patent was invalidated and two generic competitors announced launches.",
    },
    {
        "id": "s07",
        "truth": "bullish",
        "text": "The company signed a multi-year supply contract with the largest buyer in its sector.",
    },
    {
        "id": "s08",
        "truth": "bearish",
        "text": "Inventory is piling up, margins compressed 600bps, and demand guidance was withdrawn.",
    },
    {
        "id": "s09",
        "truth": "bullish",
        "text": "Same-store sales accelerated, traffic is up, and the loyalty program is growing fast.",
    },
    {
        "id": "s10",
        "truth": "bearish",
        "text": "A major data breach exposed millions of customers and class-action suits are filed.",
    },
    {
        "id": "s11",
        "truth": "bullish",
        "text": "The central bank signaled rate cuts and the company is highly rate-sensitive on the upside.",
    },
    {
        "id": "s12",
        "truth": "bearish",
        "text": "The main product was recalled for a safety defect and shipments are halted.",
    },
    {
        "id": "s13",
        "truth": "bullish",
        "text": "Order backlog hit a record and lead times extended, signaling robust demand.",
    },
    {
        "id": "s14",
        "truth": "bearish",
        "text": "A larger rival slashed prices 30% to take share in the company's core market.",
    },
    {
        "id": "s15",
        "truth": "bullish",
        "text": "Gross margins expanded for the fifth straight quarter as input costs fell.",
    },
    {
        "id": "s16",
        "truth": "bearish",
        "text": "The firm issued a going-concern warning and is in breach of its loan covenants.",
    },
    {
        "id": "s17",
        "truth": "bullish",
        "text": "A respected activist investor took a 9% stake and won three board seats.",
    },
    {
        "id": "s18",
        "truth": "bearish",
        "text": "Key engineering talent is leaving and the flagship project slipped a full year.",
    },
    # Genuinely ambiguous: mixed signals -> personas should diverge.
    {
        "id": "s19",
        "truth": "bullish",
        "text": "Earnings beat but guidance was merely in line; shares are near all-time highs.",
    },
    {
        "id": "s20",
        "truth": "bearish",
        "text": "Sales grew but margins thinned and the macro outlook for the sector is murky.",
    },
    {
        "id": "s21",
        "truth": "bullish",
        "text": "A turnaround plan is showing early traction though leverage remains elevated.",
    },
    {
        "id": "s22",
        "truth": "bearish",
        "text": "A one-time gain flattered profit while underlying organic growth stalled.",
    },
    {
        "id": "s23",
        "truth": "bullish",
        "text": "New management is cutting costs aggressively and reinvesting in a growing niche.",
    },
    {
        "id": "s24",
        "truth": "bearish",
        "text": "The stock is cheap on paper but sits in a structurally declining industry.",
    },
    # Additional clearly-bullish scenarios. The deck is deliberately NOT an even
    # 50/50 split: an exactly balanced answer key puts the true prevalence at
    # 1/2, which is a known singularity of the error-independent evaluator (its
    # accuracy equations degenerate to 0/0 there). Tilting the deck to 40/24
    # bullish (prevalence 0.625) keeps the evaluation well-conditioned while
    # still leaving 24 clear bearish scenarios, so even bullish-leaning personas
    # keep voting both ways and never collapse into a constant classifier.
    {
        "id": "s25",
        "truth": "bullish",
        "text": "A new product line sold out within hours and the company doubled its production targets.",
    },
    {
        "id": "s26",
        "truth": "bullish",
        "text": "The company won a multi-year government contract that triples its addressable market.",
    },
    {
        "id": "s27",
        "truth": "bullish",
        "text": "Subscriber growth reaccelerated and monthly churn fell to a multi-year low.",
    },
    {
        "id": "s28",
        "truth": "bullish",
        "text": "A long-running patent suit was settled in the company's favor with a large damages award.",
    },
    {
        "id": "s29",
        "truth": "bullish",
        "text": "Operating leverage kicked in: revenue rose 20% while operating expenses stayed flat.",
    },
    {
        "id": "s30",
        "truth": "bullish",
        "text": "The firm raised full-year guidance twice this quarter as bookings outpaced every forecast.",
    },
    {
        "id": "s31",
        "truth": "bullish",
        "text": "A strategic partner took a minority stake at a premium and committed to joint development.",
    },
    {
        "id": "s32",
        "truth": "bullish",
        "text": "Margins hit a record after the firm exited its lowest-margin segment and raised prices.",
    },
    # Second block (s33-s64), doubling the deck to Q=64 for a tighter
    # sampling-noise floor. The 40/24 bullish split preserves prevalence 0.625,
    # holding the problem off the 1/2 singularity (see the note above s25).
    {
        "id": "s33",
        "truth": "bullish",
        "text": "Posted record annual revenue and free cash flow and initiated its first-ever dividend.",
    },
    {
        "id": "s34",
        "truth": "bullish",
        "text": "A flagship product launch drew record preorders and sold out within a day.",
    },
    {
        "id": "s35",
        "truth": "bullish",
        "text": "Regulators granted a breakthrough designation that fast-tracks the lead program.",
    },
    {
        "id": "s36",
        "truth": "bullish",
        "text": "Closed an all-cash, immediately accretive acquisition that added no new debt.",
    },
    {
        "id": "s37",
        "truth": "bullish",
        "text": "Its credit rating was upgraded to investment grade and borrowing costs fell sharply.",
    },
    {
        "id": "s38",
        "truth": "bullish",
        "text": "Signed its largest-ever enterprise contract, a multi-year committed deal.",
    },
    {
        "id": "s39",
        "truth": "bullish",
        "text": "Margins expanded as a new automated plant came online ahead of schedule.",
    },
    {
        "id": "s40",
        "truth": "bullish",
        "text": "A major competitor exited the segment, leaving the firm the dominant supplier.",
    },
    {
        "id": "s41",
        "truth": "bullish",
        "text": "Recurring revenue crossed 80% of sales as customer churn hit a record low.",
    },
    {
        "id": "s42",
        "truth": "bullish",
        "text": "Won a landmark licensing deal that monetizes its previously idle patent portfolio.",
    },
    {
        "id": "s43",
        "truth": "bullish",
        "text": "Order backlog doubled year over year on accelerating enterprise demand.",
    },
    {
        "id": "s44",
        "truth": "bullish",
        "text": "Raised full-year guidance after a blowout quarter across every segment.",
    },
    {
        "id": "s45",
        "truth": "bullish",
        "text": "A cornerstone investor took a large stake at a premium to the market price.",
    },
    {
        "id": "s46",
        "truth": "bullish",
        "text": "Grew its net cash position while retiring high-coupon debt early.",
    },
    {
        "id": "s47",
        "truth": "bullish",
        "text": "International expansion beat plan and the new region turned profitable quickly.",
    },
    {
        "id": "s48",
        "truth": "bullish",
        "text": "A long-delayed product cleared its final certification with no conditions attached.",
    },
    {
        "id": "s49",
        "truth": "bearish",
        "text": "Slashed full-year guidance twice within a single month as demand evaporated.",
    },
    {
        "id": "s50",
        "truth": "bearish",
        "text": "A surprise goodwill writedown wiped out the entire quarter's profit.",
    },
    {
        "id": "s51",
        "truth": "bearish",
        "text": "Its largest customer terminated a contract and switched to a direct rival.",
    },
    {
        "id": "s52",
        "truth": "bearish",
        "text": "An accounting restatement triggered a delayed filing and an SEC inquiry.",
    },
    {
        "id": "s53",
        "truth": "bearish",
        "text": "A safety recall halted production of its top-selling product line.",
    },
    {
        "id": "s54",
        "truth": "bearish",
        "text": "Debt covenants were breached and lenders demanded immediate renegotiation.",
    },
    {
        "id": "s55",
        "truth": "bearish",
        "text": "The CEO and CFO resigned in the same week amid a board dispute.",
    },
    {
        "id": "s56",
        "truth": "bearish",
        "text": "A core patent expired and generic rivals immediately flooded the market.",
    },
    # Ambiguous block: mixed signals so the personas' dispositions diverge.
    {
        "id": "s57",
        "truth": "bullish",
        "text": "Revenue beat expectations but the firm guided conservatively citing macro caution.",
    },
    {
        "id": "s58",
        "truth": "bearish",
        "text": "Margins held up but unit volumes quietly declined for a third straight quarter.",
    },
    {
        "id": "s59",
        "truth": "bullish",
        "text": "A costly restructuring is underway but early customer cohorts show real traction.",
    },
    {
        "id": "s60",
        "truth": "bearish",
        "text": "A buyback was announced even as several insiders sold into the news.",
    },
    {
        "id": "s61",
        "truth": "bullish",
        "text": "Cash burn continues but a recent raise extends the runway by two years.",
    },
    {
        "id": "s62",
        "truth": "bearish",
        "text": "Headline growth impressed but was driven entirely by one large one-off deal.",
    },
    {
        "id": "s63",
        "truth": "bullish",
        "text": "The dividend was trimmed to fund a high-return capacity expansion.",
    },
    {
        "id": "s64",
        "truth": "bearish",
        "text": "The stock screens cheap but sits in a slowly shrinking end market.",
    },
]
