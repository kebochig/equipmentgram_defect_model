EQUIPMENT_HIERARCHY = {
    "Compact Excavator": {
        "Bobcat": {
            "models": ["E10", "E20", "E26", "E32", "E35"]
        },
        "Caterpillar": {
            "models": ["300.9D", "302CR", "303.5E2CR", "305E2CR", "307.5CR"]
        },
        "John Deere": {
            "models": ["17 G-Tier", "26 G-Tier", "35 G-Tier", "50 G-Tier", "60 G-Tier"]
        }
    }
}

COMPACT_EXCAVATOR_SECTIONS = {
    "General Appearance": [
        "Exterior Light",
        "Glass",
        "Hand Rails",
        "Paint",
        "Sheet Metal (Fiberglass Condition)"
    ],
    "Safety": [
        "Current Operator and Maintenance Manual",
        "Current Safety Manual",
        "Horn",
        "Safety Lock Out and Stop",
        "Seat Belt",
        "Swing Break",
        "Travel Alarm"
    ],
    "Control Station": [
        "Air Conditioner",
        "Dash Console",
        "Drivetrain Controls",
        "Engine Oil Pressure",
        "Gauges",
        "Heater",
        "Hour Meter",
        "Hydraulic Controls",
        "Limited Function Check (Control Station)",
        "Mirrors",
        "Seats ands Armrest",
        "Steering Controls",
        "Warning Light"
    ],
    "Engine": [
        "AC Compressor",
        "Blow By (Subjective Observation)",
        "Cooling System Leaks",
        "Emission Label",
        "Engine - Left Side",
        "Engine - Right Side",
        "Exhaust System",
        "Fuel Leaks",
        "Limited Function Check (Engine)",
        "Oil Leaks",
        "Radiator",
        "Starter"
    ],
    "Drivetrain": [
        "Left Drive Motor",
        "Left Final Drive",
        "Limited Function Check (Drivetrain)",
        "Right Motor Drive",
        "Right Final Drive"
    ],
    "Hydraulics": [
        "Auxiliary Hydraulic Plumbing",
        "Blade Lift Cylinder",
        "Boom Lift Cylinder(s)",
        "Boom Swing Cylinders",
        "Bucket Cylinder",
        "Hose (Hydraulics)",
        "Hydraulic Center Swivel",
        "Hydraulic Control Pattern Changer",
        "Hydraulic Tank",
        "Limited Function Check (Hydrualics)",
        "Pump (Hydraulics)",
        "Stick Cylinder",
        "Swing Motor",
        "Valves"
    ],
    "Boom Condition": [
        "Boom Condition",
        "Stick Condition",
        "Swing Tower Pivot",
        "Boom Base Pin and Bushings",
        "PIn and Bushings Boom to Stick",
        "Pin and Bushings Stick to Coupler",
        "Turntable Bearing",
        "Bottom Covers",
        "Limited Function Check"
    ],
    "Undercarriage": [
        "Left Front Idler",
        "Left Grouser Height",
        "Left Roller Frame",
        "Left Rubber Belt Drive Lugs",
        "Left Sprocket",
        "Left Track Belt Condition",
        "Left Track Rollers",
        "Left Track Tensioner",
        "Right Front Idler",
        "Right Grouser Height",
        "Right Roller Frame",
        "Right Rubber Belt Drive Lugs",
        "Right Sprocket",
        "Right Track Belt Condition",
        "Right Track Rollers",
        "Right Track Tensioners"
    ],
    "Speciality": [
        "Blade Condition",
        "Blade Cutting Edge Condition",
        "Cutting Edge Teeth Adapter",
        "Excavator Bucket Condition"
    ]
}


def validate_hierarchy(
    equipment_type: str, manufacturer: str, model: str,
    section: str, component: str
) -> tuple[bool, str]:
    from urllib.parse import unquote
    equipment_type = unquote(equipment_type).strip()
    manufacturer = unquote(manufacturer).strip()
    model = unquote(model).strip()
    section = unquote(section).strip()
    component = unquote(component).strip()

    if equipment_type not in EQUIPMENT_HIERARCHY:
        return False, f"Invalid equipment type. Must be one of: {list(EQUIPMENT_HIERARCHY.keys())}"

    if manufacturer not in EQUIPMENT_HIERARCHY[equipment_type]:
        valid = list(EQUIPMENT_HIERARCHY[equipment_type].keys())
        return False, f"Invalid manufacturer for {equipment_type}. Must be one of: {valid}"

    valid_models = EQUIPMENT_HIERARCHY[equipment_type][manufacturer]["models"]
    if model not in valid_models:
        return False, f"Invalid model for {manufacturer}. Must be one of: {valid_models}"

    if equipment_type == "Compact Excavator":
        if section not in COMPACT_EXCAVATOR_SECTIONS:
            return False, f"Invalid section. Must be one of: {list(COMPACT_EXCAVATOR_SECTIONS.keys())}"
        if component not in COMPACT_EXCAVATOR_SECTIONS[section]:
            return False, f"Invalid component for section '{section}'. Must be one of: {COMPACT_EXCAVATOR_SECTIONS[section]}"

    return True, "Valid"
