#!/usr/bin/env python3
"""
Generate scenarios.json for the PIT web app.

Produces 523 scenarios across 6 categories:
  - suspect_roleplay: 200
  - witness_roleplay: 60
  - peace_knowledge: 80
  - assessment: 120
  - scenario_presentation: 33
  - special_procedures: 30

Usage:
  python pit-app-v2/scripts/generate_scenarios.py
"""

import json
import random
import sys
import os
import hashlib

# Add project root to path so we can import scripts.pit_sft
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.pit_sft.pools import (
    OFFENCES, BEHAVIOURS, WITNESS_TYPES,
    pick_name, pick_age, pick_location, pick_datetime,
    pick_solicitor, pick_town,
)
from scripts.pit_sft.scenarios import create_scenario
from scripts.pit_sft.gen_knowledge import (
    _caution_qa, _special_warnings_qa, _pace_code_c_qa,
    _appropriate_adult_qa, _legal_advice_qa, _recording_qa,
    _questioning_qa, _peace_phases_qa, _disclosure_qa,
    _interview_planning_qa,
)
from scripts.pit_sft.gen_special import (
    _no_comment_handling, _pre_prepared_statements,
    _solicitor_interventions, _appropriate_adult_procedures,
    _interpreter_procedures, _vulnerability_recognition,
    _special_warning_delivery,
)

SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenarios.json")


# ── Difficulty mapping ────────────────────────────────────────────────────────

BEHAVIOUR_DIFFICULTY = {
    "cooperative": "beginner",
    "nervous": "beginner",
    "hostile": "intermediate",
    "deceptive": "intermediate",
    "no_comment": "intermediate",
    "vulnerable": "intermediate",
    "pre_prepared_statement": "advanced",
    "solicitor_advised_silence": "advanced",
}

KNOWLEDGE_TOPIC_DIFFICULTY = {
    "caution": "beginner",
    "recording": "beginner",
    "questioning": "beginner",
    "peace": "beginner",
    "planning": "intermediate",
    "pace": "intermediate",
    "legal_advice": "intermediate",
    "disclosure": "intermediate",
    "appropriate_adult": "intermediate",
    "special_warnings": "advanced",
}

SPECIAL_TOPIC_DIFFICULTY = {
    "no_comment": "intermediate",
    "prepared_statement": "intermediate",
    "solicitor": "intermediate",
    "appropriate_adult": "intermediate",
    "interpreter": "intermediate",
    "vulnerability": "advanced",
    "special_warning_delivery": "advanced",
}

ASSESSMENT_GRADE_DIFFICULTY = {
    "Outstanding": "advanced",
    "Good": "intermediate",
    "Satisfactory": "beginner",
    "Unsatisfactory": "beginner",
}


def make_id(mode: str, idx: int) -> str:
    return f"{mode}_{idx:04d}"


# ── Suspect roleplay scenarios (200) ──────────────────────────────────────────

def generate_suspect_roleplay(rng: random.Random) -> list:
    scenarios = []
    idx = 0

    offence_keys = list(OFFENCES.keys())
    behaviour_keys = list(BEHAVIOURS.keys())

    # 14 offences * 8 behaviours = 112 unique combos. Do 2 passes to reach 200.
    combos = [(o, b) for o in offence_keys for b in behaviour_keys]
    combos = combos + combos  # second pass generates different random details
    rng.shuffle(combos)

    for offence_key, behaviour_key in combos:
        if idx >= 200:
            break

        s = create_scenario(offence_key, behaviour_key, rng, idx)
        difficulty = BEHAVIOUR_DIFFICULTY.get(behaviour_key, "intermediate")

        scenario_json = {
            "id": make_id("suspect_roleplay", idx),
            "mode": "suspect_roleplay",
            "difficulty": difficulty,
            "offence": s.offence_label.title(),
            "offenceKey": s.offence_key,
            "statute": s.offence_statute,
            "offenceDesc": s.offence_desc,
            "suspectName": s.suspect_name,
            "suspectAge": s.suspect_age,
            "suspectGender": s.suspect_gender,
            "behaviour": s.suspect_behaviour,
            "behaviourLabel": BEHAVIOURS[behaviour_key]["label"].title(),
            "behaviourDetail": _get_behaviour_detail(s),
            "whatHappened": s.key_facts.get("what_happened", ""),
            "evidence": s.evidence_items,
            "pointsToProve": s.points_to_prove,
            "suspectDetails": _build_suspect_details(s),
            "suspectVersion": s.suspect_version.get("account", ""),
            "solicitor": s.solicitor_name if s.solicitor_present else None,
            "s36Items": s.s36_items,
            "s37Applicable": s.s37_applicable,
            "description": _build_suspect_description(s),
        }
        scenarios.append(scenario_json)
        idx += 1

    return scenarios[:200]


def _get_behaviour_detail(s) -> str:
    """Mirror the behaviour detail from prompts.py."""
    b = s.suspect_behaviour
    if b == "cooperative":
        return "You are willing to talk and answer questions. You may show remorse. Keep your answers honest but you might minimise your involvement slightly."
    elif b == "hostile":
        return "You are angry and confrontational. You deny everything aggressively. You may insult the officer or blame the victim. You are dismissive and sarcastic."
    elif b == "no_comment":
        return "You respond 'No comment' to every question. You do not elaborate or engage. You look at your solicitor before each answer."
    elif b == "deceptive":
        return f"You provide a false version of events. Your false story: {s.suspect_version.get('account', '')}. You are confident at first but become hesitant if challenged with evidence."
    elif b == "vulnerable":
        return "You struggle to understand questions. You give short, uncertain answers. You may ask for things to be repeated. You might agree with suggestions even if they are wrong because you want the interview to end."
    elif b == "nervous":
        return "You speak quickly and fidget. Your answers are rambling and jump between topics. You provide partial information with gaps."
    elif b == "pre_prepared_statement":
        return "Your solicitor will read a pre-prepared statement at the start. After that, you respond 'No comment' to all questions."
    elif b == "solicitor_advised_silence":
        return "On your solicitor's advice, you respond 'No comment' to all questions. Your solicitor may intervene to object to certain questions."
    return ""


def _build_suspect_details(s) -> str:
    parts = []
    if s.vulnerability_flags:
        parts.append(f"Has {s.vulnerability_flags[0]}.")
    if s.appropriate_adult:
        parts.append("Appropriate adult present.")
    if not parts:
        parts.append("No significant vulnerabilities identified.")
    return " ".join(parts)


def _build_suspect_description(s) -> str:
    """Short card description for the UI."""
    behaviour = BEHAVIOURS[s.suspect_behaviour]
    beh_desc = behaviour["description"]
    return f"{s.offence_desc.capitalize()} scenario. Suspect is {beh_desc.lower()}"


# ── Witness roleplay scenarios (60) ──────────────────────────────────────────

def generate_witness_roleplay(rng: random.Random) -> list:
    scenarios = []
    idx = 0

    offence_keys = list(OFFENCES.keys())
    witness_type_keys = list(WITNESS_TYPES.keys())

    # ~60 / (14 offences) = ~4 per offence, cycling through witness types
    combos = []
    for o in offence_keys:
        for wt in witness_type_keys:
            combos.append((o, wt))
    rng.shuffle(combos)

    for offence_key, witness_type_key in combos:
        if idx >= 60:
            break

        behaviour = rng.choice(["cooperative", "deceptive", "hostile"])
        s = create_scenario(offence_key, behaviour, rng, idx)
        wt = WITNESS_TYPES[witness_type_key]

        witness_gender = rng.choice(["male", "female"])
        witness_name = pick_name(rng, witness_gender)

        # Difficulty based on witness type
        if witness_type_key in ("bystander_clear", "victim_emotional"):
            difficulty = "beginner"
        elif witness_type_key in ("victim_angry", "bystander_partial", "significant"):
            difficulty = "intermediate"
        else:  # reluctant
            difficulty = "advanced"

        scenario_json = {
            "id": make_id("witness_roleplay", idx),
            "mode": "witness_roleplay",
            "difficulty": difficulty,
            "offence": s.offence_label.title(),
            "offenceKey": s.offence_key,
            "witnessName": witness_name,
            "witnessGender": witness_gender,
            "witnessType": witness_type_key,
            "witnessTypeLabel": wt["label"].title(),
            "demeanour": wt["demeanour"][0] if isinstance(wt["demeanour"], list) else wt["demeanour"],
            "recallQuality": wt["recall_quality"],
            "whatHappened": s.key_facts.get("what_happened", ""),
            "evidence": s.evidence_items,
            "description": f"Interview a {wt['label']} about {s.offence_desc}. {wt['description']}",
        }
        scenarios.append(scenario_json)
        idx += 1

    return scenarios[:60]


# ── PEACE Knowledge scenarios (80) ───────────────────────────────────────────

def generate_peace_knowledge(rng: random.Random) -> list:
    scenarios = []

    topic_funcs = {
        "caution": _caution_qa,
        "special_warnings": _special_warnings_qa,
        "pace": _pace_code_c_qa,
        "appropriate_adult": _appropriate_adult_qa,
        "legal_advice": _legal_advice_qa,
        "recording": _recording_qa,
        "questioning": _questioning_qa,
        "peace": _peace_phases_qa,
        "disclosure": _disclosure_qa,
        "planning": _interview_planning_qa,
    }

    idx = 0
    for topic_key, func in topic_funcs.items():
        qa_list = func()
        for qa in qa_list:
            difficulty = KNOWLEDGE_TOPIC_DIFFICULTY.get(qa["topic"], "intermediate")
            scenario_json = {
                "id": make_id("peace_knowledge", idx),
                "mode": "peace_knowledge",
                "difficulty": difficulty,
                "topic": qa["topic"],
                "topicLabel": qa["topic"].replace("_", " ").title(),
                "title": qa["q"],
                "fullAnswer": qa["a"],
                "description": qa["a"][:150] + "..." if len(qa["a"]) > 150 else qa["a"],
            }
            scenarios.append(scenario_json)
            idx += 1

    rng.shuffle(scenarios)
    return scenarios[:80]


# ── Assessment scenarios (120) ────────────────────────────────────────────────

def generate_assessment(rng: random.Random) -> list:
    scenarios = []
    idx = 0

    offence_keys = list(OFFENCES.keys())
    grades = ["Outstanding", "Good", "Satisfactory", "Unsatisfactory"]

    assessment_types = [
        "Full PEACE framework assessment",
        "Questioning technique assessment",
        "Caution delivery assessment",
        "Evidence presentation assessment",
        "Rapport and engagement assessment",
        "Legal compliance assessment",
        "Challenge technique assessment",
        "Closure and evaluation assessment",
    ]

    for offence_key in offence_keys:
        offence = OFFENCES[offence_key]
        for grade in grades:
            for _ in range(1):  # 1 per combo = 14 * 4 = 56 base
                if idx >= 120:
                    break
                assess_type = rng.choice(assessment_types)
                difficulty = ASSESSMENT_GRADE_DIFFICULTY.get(grade, "intermediate")
                scenario_json = {
                    "id": make_id("assessment", idx),
                    "mode": "assessment",
                    "difficulty": difficulty,
                    "offence": offence["label"].title(),
                    "offenceKey": offence_key,
                    "grade": grade,
                    "title": f"{assess_type} - {offence['label'].title()}",
                    "description": f"Paste your interview transcript for a {offence['description']} case to receive a {grade}-level assessment against PIP Level 1 standards.",
                }
                scenarios.append(scenario_json)
                idx += 1

    # Fill remaining with duplicates of different assessment types
    while len(scenarios) < 120:
        offence_key = rng.choice(offence_keys)
        offence = OFFENCES[offence_key]
        grade = rng.choice(grades)
        assess_type = rng.choice(assessment_types)
        difficulty = ASSESSMENT_GRADE_DIFFICULTY.get(grade, "intermediate")
        scenario_json = {
            "id": make_id("assessment", idx),
            "mode": "assessment",
            "difficulty": difficulty,
            "offence": offence["label"].title(),
            "offenceKey": offence_key,
            "grade": grade,
            "title": f"{assess_type} - {offence['label'].title()}",
            "description": f"Paste your interview transcript for a {offence['description']} case to receive a {grade}-level assessment against PIP Level 1 standards.",
        }
        scenarios.append(scenario_json)
        idx += 1

    rng.shuffle(scenarios)
    return scenarios[:120]


# ── Scenario Presentation (33) ───────────────────────────────────────────────

def generate_scenario_presentation(rng: random.Random) -> list:
    scenarios = []
    idx = 0

    offence_keys = list(OFFENCES.keys())

    # ~2-3 per offence type (14 offences * 2-3 = 28-42, trim to 33)
    for offence_key in offence_keys:
        count = 3 if idx + 3 <= 40 else 2
        if idx >= 40:
            break
        for _ in range(count):
            if idx >= 40:
                break
            behaviour = rng.choice(["cooperative", "hostile", "no_comment", "deceptive"])
            s = create_scenario(offence_key, behaviour, rng, idx)
            offence = OFFENCES[offence_key]

            difficulty = BEHAVIOUR_DIFFICULTY.get(behaviour, "intermediate")

            scenario_json = {
                "id": make_id("scenario_presentation", idx),
                "mode": "scenario_presentation",
                "difficulty": difficulty,
                "offence": offence["label"].title(),
                "offenceKey": offence_key,
                "title": f"{offence['label'].title()} scenario briefing",
                "description": f"Get a realistic training briefing for a {offence['description']} interview. Includes suspect details, evidence summary, and interview planning guidance.",
            }
            scenarios.append(scenario_json)
            idx += 1

    rng.shuffle(scenarios)
    return scenarios[:33]


# ── Special Procedures (30) ──────────────────────────────────────────────────

def generate_special_procedures(rng: random.Random) -> list:
    scenarios = []

    topic_funcs = {
        "no_comment": _no_comment_handling,
        "prepared_statement": _pre_prepared_statements,
        "solicitor": _solicitor_interventions,
        "appropriate_adult": _appropriate_adult_procedures,
        "interpreter": _interpreter_procedures,
        "vulnerability": _vulnerability_recognition,
        "special_warning_delivery": _special_warning_delivery,
    }

    idx = 0
    for topic_key, func in topic_funcs.items():
        examples = func(rng)
        for ex in examples:
            q = ex["conversations"][1]["value"]
            a = ex["conversations"][2]["value"]
            difficulty = SPECIAL_TOPIC_DIFFICULTY.get(topic_key, "intermediate")
            scenario_json = {
                "id": make_id("special_procedures", idx),
                "mode": "special_procedures",
                "difficulty": difficulty,
                "topic": topic_key,
                "topicLabel": topic_key.replace("_", " ").title(),
                "title": q,
                "fullAnswer": a,
                "description": a[:150] + "..." if len(a) > 150 else a,
            }
            scenarios.append(scenario_json)
            idx += 1

    rng.shuffle(scenarios)
    return scenarios[:30]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)

    print("Generating scenarios...")

    suspects = generate_suspect_roleplay(rng)
    print(f"  suspect_roleplay: {len(suspects)}")

    witnesses = generate_witness_roleplay(rng)
    print(f"  witness_roleplay: {len(witnesses)}")

    knowledge = generate_peace_knowledge(rng)
    print(f"  peace_knowledge: {len(knowledge)}")

    assessments = generate_assessment(rng)
    print(f"  assessment: {len(assessments)}")

    presentations = generate_scenario_presentation(rng)
    print(f"  scenario_presentation: {len(presentations)}")

    specials = generate_special_procedures(rng)
    print(f"  special_procedures: {len(specials)}")

    all_scenarios = suspects + witnesses + knowledge + assessments + presentations + specials
    total = len(all_scenarios)
    print(f"\nTotal: {total}")

    output = {
        "version": 1,
        "total": total,
        "scenarios": all_scenarios,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Written to {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
