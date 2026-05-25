"""
CLI runner for the scraper agent.

Usage examples:

  # Scrape one component
  python -m scraper_agent.run --component "Boom Lift Cylinder(s)" --section Hydraulics

  # Scrape all components in a section
  python -m scraper_agent.run --section Hydraulics

  # Scrape everything (all sections, all components)
  python -m scraper_agent.run --all

  # Control how many images per component (default: 5)
  python -m scraper_agent.run --section Undercarriage --count 8

  # Show dataset stats
  python -m scraper_agent.run --stats
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running from the defect_model root
sys.path.insert(0, str(Path(__file__).parent.parent))

from equipment import COMPACT_EXCAVATOR_SECTIONS
from scraper_agent.agent import run_agent
from scraper_agent.tools import DATASET_DIR, LABELS_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("scraper_agent.run")

# Delay between components to avoid rate limits
INTER_COMPONENT_DELAY_SEC = 10


def print_stats():
    """Print a summary of the current dataset."""
    if not LABELS_FILE.exists():
        print("No dataset found yet.")
        return

    entries = [json.loads(line) for line in LABELS_FILE.open()]
    if not entries:
        print("Dataset is empty.")
        return

    total = len(entries)
    by_component: dict[str, int] = {}
    by_confidence: dict[str, int] = {}
    by_fault: dict[str, int] = {}

    for e in entries:
        comp = e.get("component", "unknown")
        by_component[comp] = by_component.get(comp, 0) + 1
        conf = e.get("label_confidence", "unknown")
        by_confidence[conf] = by_confidence.get(conf, 0) + 1
        fault = e.get("fault_type", "unknown")
        by_fault[fault] = by_fault.get(fault, 0) + 1

    print(f"\n{'='*50}")
    print(f"DATASET STATS  —  {DATASET_DIR}")
    print(f"{'='*50}")
    print(f"Total entries : {total}")
    print(f"\nBy confidence :")
    for k, v in sorted(by_confidence.items()):
        print(f"  {k:10s}: {v}")
    print(f"\nTop fault types:")
    for fault, count in sorted(by_fault.items(), key=lambda x: -x[1])[:10]:
        print(f"  {fault:30s}: {count}")
    print(f"\nComponents covered: {len(by_component)} / {sum(len(v) for v in COMPACT_EXCAVATOR_SECTIONS.values())}")
    print(f"{'='*50}\n")


def scrape_component(component: str, section: str, count: int):
    logger.info("Scraping: %s > %s (target=%d)", section, component, count)
    result = run_agent(component=component, section=section, target_count=count)
    logger.info(
        "Done: %s — saved=%d downloaded=%d/%d iterations=%d",
        component,
        result["labels_saved"],
        result["downloads_succeeded"],
        result["downloads_attempted"],
        result["iterations"],
    )
    return result


def already_scraped(component: str, min_entries: int = 3) -> bool:
    """Return True if this component already has enough entries in the dataset."""
    if not LABELS_FILE.exists():
        return False
    count = sum(
        1 for line in LABELS_FILE.open()
        if json.loads(line).get("component") == component
    )
    return count >= min_entries


def main():
    parser = argparse.ArgumentParser(description="Scraper agent for faulty equipment images")
    parser.add_argument("--component", help="Specific component name to scrape")
    parser.add_argument("--section", help="Section name — scrapes all components in section")
    parser.add_argument("--all", action="store_true", help="Scrape all components")
    parser.add_argument("--count", type=int, default=5, help="Target images per component (default: 5)")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics and exit")
    parser.add_argument("--resume", action="store_true", help="Skip components that already have enough entries")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if not (args.component or args.section or args.all):
        parser.print_help()
        return

    # Build the work list
    work: list[tuple[str, str]] = []  # (component, section)

    if args.component:
        if not args.section:
            # Try to auto-detect section
            for sec, comps in COMPACT_EXCAVATOR_SECTIONS.items():
                if args.component in comps:
                    args.section = sec
                    break
            if not args.section:
                logger.error("Could not find section for component '%s'. Use --section.", args.component)
                sys.exit(1)
        work.append((args.component, args.section))

    elif args.section:
        if args.section not in COMPACT_EXCAVATOR_SECTIONS:
            logger.error("Unknown section '%s'. Valid: %s", args.section, list(COMPACT_EXCAVATOR_SECTIONS.keys()))
            sys.exit(1)
        work = [(comp, args.section) for comp in COMPACT_EXCAVATOR_SECTIONS[args.section]]

    elif args.all:
        for section, components in COMPACT_EXCAVATOR_SECTIONS.items():
            for comp in components:
                work.append((comp, section))

    # Filter already-scraped if resuming
    if args.resume:
        before = len(work)
        work = [(c, s) for c, s in work if not already_scraped(c, min_entries=args.count)]
        logger.info("Resume: skipping %d already-scraped components, %d remaining", before - len(work), len(work))

    if not work:
        logger.info("Nothing to scrape.")
        print_stats()
        return

    logger.info("Scraping %d component(s), %d images each", len(work), args.count)

    all_results = []
    for i, (component, section) in enumerate(work):
        result = scrape_component(component, section, args.count)
        all_results.append(result)

        if i < len(work) - 1:
            logger.info("Waiting %ds before next component...", INTER_COMPONENT_DELAY_SEC)
            time.sleep(INTER_COMPONENT_DELAY_SEC)

    # Summary
    total_saved = sum(r["labels_saved"] for r in all_results)
    logger.info("All done. Total labels saved this run: %d", total_saved)
    print_stats()


if __name__ == "__main__":
    main()
