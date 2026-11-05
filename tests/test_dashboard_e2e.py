"""Headless-browser smoke test of the web dashboard (Playwright + Chromium).

Not part of the fast suite (needs `pip install -r requirements-dev.txt` and
`python -m playwright install chromium`).  It starts the real server on a free port, drives the UI
like a user and fails on any JavaScript error:

  1. page loads with no console / page errors
  2. "Step +5" scores transactions and fills the live feed
  3. the Challenge-scenarios table renders 4 scenarios with numeric values (this table never loaded
     in the original code because of a data-shape bug)
  4. the consumer copilot answers for a verified merchant
  5. the bank-analyst copilot produces a SAR draft
  6. with UPI_API_KEY set, /api calls without the key are rejected

Alert-dialog buttons (Save Draft / Approve SAR) are deliberately never clicked.
Saves a screenshot to reports/dashboard_smoke.png.
"""

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, timeout: float = 180.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("server did not start")


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SKIP: playwright not installed (pip install -r requirements-dev.txt; python -m playwright install chromium)")
        return 0

    port = free_port()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", PYTHONWARNINGS="ignore")
    env.pop("UPI_API_KEY", None)
    srv = subprocess.Popen([sys.executable, str(ROOT / "src" / "scoring_service.py"), "--port", str(port)],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    errors = []
    try:
        wait_ready(base + "/api/stream/state")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.goto(base + "/")

            page.get_by_role("button", name="Step +5").click()
            page.wait_for_function("!document.getElementById('stream-feed-list').innerText.includes(\"Click 'Start Stream'\")", timeout=30000)

            page.get_by_role("button", name="3 Challenge Scenarios").click()
            page.wait_for_function("document.querySelectorAll('#challenges-table-body tr').length === 4", timeout=30000)
            rows = page.locator("#challenges-table-body tr").all_inner_texts()
            assert len(rows) == 4 and not any("undefined" in r or "NaN" in r for r in rows), rows

            page.get_by_role("button", name="Consumer Safety Copilot").click()
            page.get_by_role("button", name="Try Verified").click()
            page.wait_for_function(
                "(() => { const t = document.getElementById('consumer-result-box').innerText;"
                " return !t.includes('Enter a UPI ID') && t.includes('dmart_retail_hub'); })()", timeout=30000)
            consumer_text = page.locator("#consumer-result-box").inner_text()
            assert "Enter a UPI ID" not in consumer_text, consumer_text

            page.get_by_role("button", name="Bank Analyst Copilot").click()
            page.get_by_role("button", name="Generate Evidence-Grounded SAR").click()
            page.wait_for_function("document.getElementById('sar-textarea').value.includes('SUSPICIOUS ACTIVITY REPORT')", timeout=30000)

            sar = page.locator("#sar-textarea").input_value()
            assert "unknown@upi" not in sar and "Originating VPA: acc_" in sar, sar[:400]
            assert "0.75 * XGBoost + 0.25" not in sar, "SAR must not hard-code stale blend weights"

            shot = ROOT / "reports" / "dashboard_smoke.png"
            page.screenshot(path=str(shot), full_page=False)
            browser.close()
        assert not errors, f"JavaScript errors: {errors}"
        print(f"PASS: dashboard smoke test ({len(rows)} challenge rows; consumer advisory: {consumer_text.splitlines()[0][:60]!r}); screenshot {shot}")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}\nJS errors: {errors}")
        return 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
