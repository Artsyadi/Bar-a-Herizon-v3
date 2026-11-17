"""Freeze v3 = v2 + the rules of config/v3_post.json: write models/v3/manifest.json.

No model is trained or copied: v3 uses models/v2 as they are (their own manifest is checked first).
make_submission.py --post v3 and step8_bar_ath.py --post v3 refuse to run if any file listed here
changed afterwards.

    BH_MODEL_VERSION=v2 python scripts/v3_freeze.py
"""

import datetime
import hashlib
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402

FILES = ["models/v2/manifest.json", "config/v3_post.json", "bh/v3post.py",
         "scripts/make_submission.py", "scripts/step8_bar_ath.py"]


def md5(rel):
    with open(os.path.join(config.ROOT, rel), "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main():
    v2 = json.load(open(os.path.join(config.PROJECT, "models", "v2", "manifest.json"), encoding="utf-8"))
    bad = [rel for rel, h in v2["md5"].items() if md5(rel) != h]
    if bad:
        raise SystemExit("v2 files changed since the v2 freeze: {}".format(bad))
    out = os.path.join(config.PROJECT, "models", "v3")
    os.makedirs(out, exist_ok=True)
    man = {
        "version": "v3 = v2 models + config/v3_post.json rules (no retraining)",
        "base": "v2",
        "frozen": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dev_result": "30 IDSSE dev units: v2 0.6135 -> v3 0.6332 (+0.0197, 6 of 6 matches better, worst +0.0074); scripts/v3_post_check.py",
        "never_used_for_choices": v2["never_used_for_training_or_choices"],
        "python": platform.python_version(),
        "md5": {rel: md5(rel) for rel in FILES},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
        fh.write("\n")
    print("v2 manifest ({} files) unchanged: PASS".format(len(v2["md5"])))
    print("wrote models/v3/manifest.json ({} files)".format(len(FILES)))


if __name__ == "__main__":
    main()
