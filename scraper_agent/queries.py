"""
Search query templates per component section.
Maps equipment sections to the fault types most likely to produce
useful image results for that section.
"""

from typing import NamedTuple


class QuerySet(NamedTuple):
    fault_types: list[str]
    search_templates: list[str]


# Fault types relevant to each section
SECTION_FAULT_FOCUS = {
    "General Appearance": [
        "crack", "corrosion", "paint damage", "dent", "deformation"
    ],
    "Safety": [
        "damage", "broken", "missing", "defective"
    ],
    "Control Station": [
        "damage", "broken", "cracked", "worn"
    ],
    "Engine": [
        "oil leak", "coolant leak", "crack", "corrosion", "damage", "blown seal"
    ],
    "Drivetrain": [
        "oil leak", "damage", "crack", "worn", "failure"
    ],
    "Hydraulics": [
        "hydraulic leak", "seal failure", "hose damage", "crack", "corrosion",
        "cylinder damage", "blown seal", "oil leak"
    ],
    "Boom Condition": [
        "crack", "structural damage", "pin wear", "bushing wear",
        "weld failure", "corrosion", "deformation"
    ],
    "Undercarriage": [
        "track wear", "grouser wear", "sprocket damage", "idler damage",
        "roller damage", "crack", "corrosion", "track damage"
    ],
    "Speciality": [
        "crack", "wear", "damage", "broken", "bent"
    ],
}

# Generic fault terms applied to any component when section-specific list is short
GENERIC_FAULT_TERMS = [
    "crack", "damage", "wear", "corrosion", "failure", "defect", "broken"
]


def build_queries(component: str, section: str) -> list[dict]:
    """
    Generate a list of search query dicts for a given component/section.
    Each dict has: query (str), expected_fault (str).
    """
    faults = SECTION_FAULT_FOCUS.get(section, GENERIC_FAULT_TERMS)
    queries = []

    for fault in faults:
        # Multiple query patterns increase hit diversity
        patterns = [
            f"excavator {component} {fault}",
            f"compact excavator {component} {fault} inspection",
            f"mini excavator {component} {fault} damage",
            f"{component} {fault} heavy equipment",
        ]
        for pattern in patterns[:2]:  # 2 patterns per fault to keep volume manageable
            queries.append({
                "query": pattern,
                "expected_fault": fault,
                "component": component,
                "section": section,
            })

    return queries
