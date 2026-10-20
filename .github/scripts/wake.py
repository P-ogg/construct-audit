"""Open the Streamlit app; if it is asleep, press the wake-up button and wait until the app has loaded."""
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1]
MARKER = "Audit your articles"      # the app's first tab; change it here if the app's wording changes
with sync_playwright() as p:
    page = p.chromium.launch().new_page()
    page.goto(URL, timeout=90_000)
    page.wait_for_timeout(8_000)
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count():
        print("app was asleep: waking it")
        wake.first.click()
    # the app runs inside an iframe; wait for its title to appear
    def loaded():
        for f in page.frames:
            try:
                if MARKER in f.content():
                    return True
            except Exception:           # a frame can detach while the app reloads
                pass
        return False

    for _ in range(36):
        if loaded():
            print("app is awake"); break
        page.wait_for_timeout(5_000)
    else:
        print("frames seen:", [f.url for f in page.frames])
        sys.exit(f"the app did not show '{MARKER}' within 3 minutes")
