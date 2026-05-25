"""
Claude subagent for scraping and labeling faulty equipment component images.

Flow per component:
  1. Claude uses web_search (server-side) to find image candidates
  2. Claude uses web_fetch (server-side) to read pages and extract context
  3. Claude calls download_image (client-side) to save the image locally
  4. Claude calls save_to_dataset (client-side) to write the ground truth label

Labels come entirely from human-written text on the source page
(alt text, captions, article text) — not from any model vision pass.
This makes the dataset ground truth.
"""

import json
import logging
import os
import time

import anthropic

from .tools import CUSTOM_TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger("scraper_agent.agent")

# ---------------------------------------------------------------------------
# Tool list: server-side web tools + our custom client-side tools
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    # Anthropic-hosted — Claude calls them, Anthropic executes them
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "web_fetch_20260209", "name": "web_fetch"},
    # Client-side — we execute these when Claude calls them
    *CUSTOM_TOOL_DEFINITIONS,
]

SYSTEM_PROMPT = """You are a research agent building a ground truth dataset of faulty heavy equipment components for an inspection AI system.

Your task for this session: find and label {target_count} images of faulty '{component}' components from compact excavators.

STRICT LABELING RULE — THIS IS CRITICAL:
The label (fault_description and page_context) MUST come verbatim from text on the web page.
Do NOT use your own assessment of what you see in the image.
Do NOT infer faults from context not present on the page.
The label is what the human author of the page wrote — that is the ground truth.

WORKFLOW:
1. Use web_search to find images: try queries like "excavator {component} crack", "compact excavator {component} hydraulic leak", "{component} damaged excavator inspection", etc.
2. Use web_fetch on promising result pages to read the full content — look for:
   - Image alt text that describes a fault
   - Image captions near the photo
   - Article text explaining what the fault is
3. When you find a page with a clear image URL + fault description:
   a. Call download_image with the direct image URL
   b. If download succeeds, call save_to_dataset with the image path AND the verbatim text from the page
4. Vary your fault queries — cover: crack, corrosion, hydraulic leak, wear, deformation, damage, broken seal, etc.

CONFIDENCE LEVELS:
- high: page has explicit caption or alt-text naming the exact fault on this image
- medium: article text clearly describes this fault type, image is in that context
- low: fault type inferred from the article topic (still valid as ground truth)

SKIP if:
- No fault description text found on the page
- Image URL leads to a thumbnail (< 10KB) — the download tool will reject it
- Page is behind a paywall or login
- Image is a stock illustration, not a real equipment photo

Target: {target_count} saved entries with label_confidence of medium or high.
"""

MAX_LOOP_ITERATIONS = 30  # safety cap on the agentic loop


def run_agent(
    component: str,
    section: str,
    target_count: int = 5,
    api_key: str | None = None,
) -> dict:
    """
    Run the scraper agent for a single component.

    Returns a summary dict with counts of successes, failures, and total saved.
    """
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    system = SYSTEM_PROMPT.format(
        component=component,
        section=section,
        target_count=target_count,
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Find and label {target_count} images of faulty '{component}' components "
                f"(section: {section}, equipment: Compact Excavator). "
                f"Focus on images with clear fault descriptions on the source page. "
                f"Use varied search queries to cover different fault types."
            ),
        }
    ]

    stats = {"downloads_attempted": 0, "downloads_succeeded": 0, "labels_saved": 0}
    iteration = 0

    while iteration < MAX_LOOP_ITERATIONS:
        iteration += 1
        logger.info(
            "[%s] Loop iteration %d/%d", component, iteration, MAX_LOOP_ITERATIONS
        )

        try:
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=8000,
                thinking={"type": "adaptive"},
                system=system,
                tools=ALL_TOOLS,
                messages=messages,
            )
        except anthropic.RateLimitError:
            logger.warning("[%s] Rate limited — waiting 60s", component)
            time.sleep(60)
            continue
        except anthropic.APIStatusError as e:
            logger.error("[%s] API error: %s", component, e)
            break

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        # ----------------------------------------------------------------
        # pause_turn: server-side tool loop hit its iteration limit
        # Re-send without adding any user message — the API resumes automatically
        # ----------------------------------------------------------------
        if response.stop_reason == "pause_turn":
            logger.debug("[%s] pause_turn — resuming server-side loop", component)
            continue

        # ----------------------------------------------------------------
        # tool_use: Claude wants to call one of our custom client-side tools
        # ----------------------------------------------------------------
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info("[%s] Tool call: %s", component, block.name)
                result = execute_tool(block.name, block.input)

                # Track stats
                if block.name == "download_image":
                    stats["downloads_attempted"] += 1
                    if result.get("success"):
                        stats["downloads_succeeded"] += 1
                    else:
                        logger.warning(
                            "[%s] Download failed: %s", component, result.get("error")
                        )

                if block.name == "save_to_dataset" and result.get("success"):
                    stats["labels_saved"] += 1
                    logger.info(
                        "[%s] Saved label %d/%d",
                        component, stats["labels_saved"], target_count,
                    )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Stop early if target reached
            if stats["labels_saved"] >= target_count:
                logger.info(
                    "[%s] Target reached (%d labels). Stopping.",
                    component, stats["labels_saved"],
                )
                break

            continue

        # ----------------------------------------------------------------
        # end_turn: Claude finished
        # ----------------------------------------------------------------
        if response.stop_reason == "end_turn":
            logger.info(
                "[%s] Agent finished. labels_saved=%d downloads=%d/%d",
                component,
                stats["labels_saved"],
                stats["downloads_succeeded"],
                stats["downloads_attempted"],
            )
            break

    return {
        "component": component,
        "section": section,
        **stats,
        "iterations": iteration,
    }
