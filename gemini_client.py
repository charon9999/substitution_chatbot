import json

from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K_FINAL


client = genai.Client(api_key=GEMINI_API_KEY)

# Minimal schema — only fields Gemini must determine (not derivable from DB)
SUBSTITUTE_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "substitutes": types.Schema(
            type="ARRAY",
            items=types.Schema(
                type="OBJECT",
                properties={
                    "sku": types.Schema(type="STRING", description="Product SKU of the chosen substitute"),
                    "rank": types.Schema(type="INTEGER", description="Rank 1-5, 1 being best"),
                    "reason": types.Schema(
                        type="STRING",
                        description="Why this is a good substitute — functional match, spec comparison, value proposition",
                    ),
                    "unit_type": types.Schema(
                        type="STRING",
                        description="DIVISIBLE if the quantity unit can be scaled (sheets, feet, ml, oz) or ABSOLUTE if it cannot (tabs, compartments, drawers, holes, ports)",
                    ),
                    "qty_needed": types.Schema(
                        type="INTEGER",
                        description="How many units of the candidate the user must buy to fulfill their quantity. Rounded UP to whole number. For ABSOLUTE units where specs match, this is 1.",
                    ),
                    "comparison_notes": types.Schema(
                        type="STRING",
                        description="Step-by-step calculation: how qty_needed was determined, any unit conversions (ft->in, etc.), and why this quantity covers the user's need. For ABSOLUTE units, explain why the specs are equivalent.",
                    ),
                },
                required=["sku", "rank", "reason", "unit_type", "qty_needed", "comparison_notes"],
            ),
        ),
    },
    required=["substitutes"],
)


def rank_substitutes(
    source_item: dict,
    candidates: list[dict],
) -> dict:
    """Use Gemini with schema enforcement to rank candidates. Returns only fields Gemini must determine."""

    candidates_info = "\n\n".join(
        f"--- Candidate {i+1} (SKU: {c['sku']}) ---\n{c['document']}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are a product substitution expert for an office/industrial supply company.

A user wants to find substitutes for a product they are currently buying.

SOURCE ITEM (user-provided):
- Name: {source_item['name']}
- Description: {source_item['description']}
- Supercategory: {source_item['supercategory']}
- Category: {source_item['category']}
- Quantity Needed: {source_item['quantity']} {source_item.get('quantity_unit', '')}

CANDIDATE PRODUCTS FROM OUR CATALOG:
{candidates_info}

CRITICAL RULES FOR UNIT COMPARISON:

1. CLASSIFY each product attribute as DIVISIBLE or ABSOLUTE:
   - DIVISIBLE units CAN be scaled/split: sheets, pages, rolls, feet, inches, yards, meters, ml, oz, lbs, sq ft, etc.
   - ABSOLUTE units CANNOT be scaled: tabs (in a folder), compartments, drawers, holes, ports, pockets, dividers, slots, buttons, keys, etc.

2. For ABSOLUTE attributes:
   - A 24-tab folder is NOT comparable to a 12-tab folder by doing 24/12 ratio. They are fundamentally different products.
   - The candidate MUST have the SAME or very similar absolute spec to be a valid substitute.
   - Do NOT include candidates with mismatched absolute specs.

3. For DIVISIBLE units, calculate qty_needed (always round UP to next whole number):
   - Example: User buys 500 sheets. Candidate sells 5000 sheets/case.
     qty_needed = ceil(500 / 5000) = 1 unit needed
   - Example: User buys 2000 sheets. Candidate sells 500 sheets/ream.
     qty_needed = ceil(2000 / 500) = 4 units needed
   - For dimensional products, CONVERT to common base units first:
     * feet -> inches (x12), yards -> inches (x36), meters -> inches (x39.37), cm -> inches (x0.3937)

4. RANKING priority:
   a. Functional match (same purpose, matching absolute specs)
   b. Value (lower total spend = qty_needed * candidate_price)
   c. Brand/quality similarity

Return the top {TOP_K_FINAL} best substitutes. If fewer are suitable, return fewer. Do NOT pad with unsuitable products."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            responseMimeType="application/json",
            responseSchema=SUBSTITUTE_SCHEMA,
        ),
    )

    return json.loads(response.text)
