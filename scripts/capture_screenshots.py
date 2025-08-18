"""
Automated screenshot capture for the Streamlit app using Playwright (Chromium, headless).

Usage:
  python scripts/capture_screenshots.py --base-url http://localhost:8501 --out-dir docs/screenshots

Note: Requires `pip install playwright` and `python -m playwright install chromium`.
Selectors may need small tweaks if Streamlit updates its DOM.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page


def shot(page: Page, out_dir: Path, name: str, wait: float = 0.6):
    out_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(wait)
    path = out_dir / name
    page.screenshot(path=str(path), full_page=True)
    print("Saved", path)


def set_selectbox(page: Page, label: str, option_text: str):
    # Click on the selectbox near label, then pick option by text
    label_el = page.get_by_text(label, exact=True)
    container = label_el.locator("xpath=..")  # parent
    # The clickable widget is usually the next sibling; use a broader search
    container.locator("xpath=following::div[contains(@class,'stSelectbox')][1]").click()
    page.get_by_role("option", name=option_text).click()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8501")
    ap.add_argument("--out-dir", default="docs/screenshots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(args.base_url, wait_until="networkidle")

        # 01 Home
        shot(page, out_dir, "01_home_page.png")

        # Basic EN->HI translation
        try:
            set_selectbox(page, "Source Language", "English")
        except Exception:
            pass
        try:
            set_selectbox(page, "Target Language", "Hindi")
        except Exception:
            pass
        page.get_by_label("Source Text").fill("Hello world")
        page.get_by_role("button", name="Translate").click()
        page.wait_for_timeout(1500)
        shot(page, out_dir, "03_en_hi_basic_translation.png")

        # Metrics with reference
        page.get_by_label("Optional: Reference translation (for metrics)").fill("नमस्ते दुनिया")
        page.get_by_role("button", name="Translate").click()
        page.wait_for_timeout(1500)
        shot(page, out_dir, "04_metrics_with_reference.png")

        # Google compare expander
        try:
            page.get_by_text("Compare with Google Translate").click()
            page.wait_for_timeout(800)
            shot(page, out_dir, "05_google_compare_expanded.png")
        except Exception:
            pass

        # Transliteration auto-detect
        page.get_by_label("Source Text").fill("Namaste")
        page.get_by_role("button", name="Translate").click()
        page.wait_for_timeout(1200)
        shot(page, out_dir, "06_transliteration_auto_namaste.png")

        browser.close()


if __name__ == "__main__":
    main()
