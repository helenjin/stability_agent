"""Path bootstrap so `import exp_helpers...` resolves to the vendored, unmodified
ARES source tree, with the inert `vllm` stub available for its unused import.

Import this module (for its side effect) before importing anything from
`exp_helpers`. Every other new module in ares_topodev does this.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARES_TOPODEV_ROOT = os.path.dirname(_HERE)
STUBS_DIR = os.path.join(_ARES_TOPODEV_ROOT, "vendor", "_stubs")
ARES_SRC_DIR = os.path.join(_ARES_TOPODEV_ROOT, "vendor", "ares", "src")
ARES_DATA_DIR = os.path.join(_ARES_TOPODEV_ROOT, "vendor", "ares", "data")

for _p in (STUBS_DIR, ARES_SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# vendor/ares/src/exp_helpers/models/openai_llm.py constructs an `openai.OpenAI()`
# client at *import* time, which raises if OPENAI_API_KEY is unset (newer openai
# SDK versions require a key even to construct the client object). We only want
# real runs to require a real key; structural work (imports, unit tests, dry-run)
# should not need one. `setdefault` never overrides a real key you've already set.
os.environ.setdefault("OPENAI_API_KEY", "sk-ares-topodev-placeholder-not-a-real-key")

# vendor/ares/src/exp_helpers/models/openai_llm.py's batch_generate() has a
# leftover `import pdb; pdb.set_trace()` in its per-call exception handler. On
# a transient API error (rate limit, timeout, network blip) this would drop
# into an interactive debugger with no attached TTY in a headless run, hanging
# the process forever with no error surfaced. We neutralize just this debug
# hook from our own code (the vendored file itself is untouched) so a
# transient error is logged and the run continues instead of freezing.
import pdb  # noqa: E402

pdb.set_trace = lambda *args, **kwargs: None

