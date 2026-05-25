"""
Custom tool implementations for the scraper agent.
These run on the client side — Claude calls them, we execute them.
Labels come exclusively from web page text (alt text, captions, article text),
not from any model analyzing the image. This makes them ground truth.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger("scraper_agent.tools")

DATASET_DIR = Path(__file__).parent.parent / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_FILE = DATASET_DIR / "labels.jsonl"

DATASET_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0; +research)"
}

# ---------------------------------------------------------------------------
# download_image
# ---------------------------------------------------------------------------

def download_image(
    url: str,
    component: str,
    section: str,
    fault_type: str,
) -> dict:
    """
    Download an image from a URL and save it to the dataset images directory.
    Returns the local file path on success, or an error message.
    """
    try:
        response = requests.get(url, timeout=15, headers=HEADERS, stream=True)

        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}"}

        content_type = response.headers.get("content-type", "")
        if not any(t in content_type for t in ("image/jpeg", "image/png", "image/webp", "image/gif")):
            return {"success": False, "error": f"Not a supported image type: {content_type}"}

        img_data = response.content

        if len(img_data) < 10_000:
            return {"success": False, "error": f"Image too small ({len(img_data)} bytes) — likely a thumbnail or placeholder"}

        if len(img_data) > 20 * 1024 * 1024:
            return {"success": False, "error": "Image exceeds 20MB limit"}

        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        ext = "jpg"
        if "png" in content_type:
            ext = "png"
        elif "webp" in content_type:
            ext = "webp"

        component_slug = component.lower().replace(" ", "_").replace("/", "_")[:30]
        fault_slug = fault_type.lower().replace(" ", "_")[:20]
        filename = f"{component_slug}__{fault_slug}__{url_hash}.{ext}"
        filepath = IMAGES_DIR / filename

        filepath.write_bytes(img_data)
        logger.info("Downloaded image: %s (%d KB)", filename, len(img_data) // 1024)

        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
            "size_bytes": len(img_data),
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# save_to_dataset
# ---------------------------------------------------------------------------

def save_to_dataset(
    image_path: str,
    component: str,
    section: str,
    equipment_type: str,
    fault_type: str,
    fault_description: str,
    source_url: str,
    page_context: str,
    search_query: str,
    label_confidence: str,
) -> dict:
    """
    Append a labeled entry to the JSONL ground truth dataset.

    The label_source is always 'web_context' — the fault_description and
    page_context come from human-written text on the source page, not from
    any model inspecting the image. This is what makes it ground truth.
    """
    entry = {
        "image_path": image_path,
        "component": component,
        "section": section,
        "equipment_type": equipment_type,
        "fault_type": fault_type,
        "fault_description": fault_description,       # verbatim from page
        "source_url": source_url,
        "page_context": page_context[:600],            # surrounding text
        "search_query": search_query,
        "label_confidence": label_confidence,          # high / medium / low
        "label_source": "web_context",                 # NOT a model
        "labeled_at": datetime.now().isoformat(),
    }

    with LABELS_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    total = sum(1 for _ in LABELS_FILE.open())
    logger.info(
        "Saved label: component=%s fault=%s confidence=%s (total=%d)",
        component, fault_type, label_confidence, total,
    )

    return {"success": True, "total_dataset_entries": total}


# ---------------------------------------------------------------------------
# Tool definitions for the Anthropic API
# ---------------------------------------------------------------------------

CUSTOM_TOOL_DEFINITIONS = [
    {
        "name": "download_image",
        "description": (
            "Download an image from a direct URL and save it to the dataset. "
            "Use this after finding an image URL that shows a faulty component. "
            "Returns the local file path if successful."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Direct URL to the image file (must end in .jpg, .png, .webp, etc.)",
                },
                "component": {
                    "type": "string",
                    "description": "Component name exactly as in the equipment hierarchy",
                },
                "section": {
                    "type": "string",
                    "description": "Section name (e.g. Hydraulics, Undercarriage)",
                },
                "fault_type": {
                    "type": "string",
                    "description": "Short fault type label (e.g. crack, hydraulic_leak, corrosion, wear)",
                },
            },
            "required": ["url", "component", "section", "fault_type"],
        },
    },
    {
        "name": "save_to_dataset",
        "description": (
            "Save a downloaded image and its ground truth label to the dataset. "
            "The fault_description and page_context MUST come verbatim from text "
            "on the web page (alt text, caption, article text) — not from your own "
            "interpretation of the image. This is what makes it ground truth."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Local file path returned by download_image",
                },
                "component": {"type": "string"},
                "section": {"type": "string"},
                "equipment_type": {
                    "type": "string",
                    "description": "Always 'Compact Excavator' for this project",
                },
                "fault_type": {
                    "type": "string",
                    "description": "Short fault type (crack, leak, corrosion, wear, damage, deformation)",
                },
                "fault_description": {
                    "type": "string",
                    "description": "Exact text from the page describing the fault — alt text, caption, or article sentence",
                },
                "source_url": {
                    "type": "string",
                    "description": "URL of the page where the image was found",
                },
                "page_context": {
                    "type": "string",
                    "description": "The surrounding paragraph or article excerpt that describes the fault",
                },
                "search_query": {
                    "type": "string",
                    "description": "The search query that found this image",
                },
                "label_confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": (
                        "high: page has clear caption/alt-text naming the exact fault | "
                        "medium: article text describes the fault clearly but caption is generic | "
                        "low: fault inferred from article topic"
                    ),
                },
            },
            "required": [
                "image_path", "component", "section", "equipment_type",
                "fault_type", "fault_description", "source_url",
                "page_context", "search_query", "label_confidence",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, inputs: dict) -> dict:
    """Dispatch a tool call to its implementation."""
    if name == "download_image":
        return download_image(**inputs)
    elif name == "save_to_dataset":
        return save_to_dataset(**inputs)
    else:
        return {"success": False, "error": f"Unknown tool: {name}"}
