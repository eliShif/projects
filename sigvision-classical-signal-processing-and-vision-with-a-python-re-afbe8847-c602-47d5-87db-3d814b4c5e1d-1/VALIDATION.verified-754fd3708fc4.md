# Reproducible validation

Verified at 2026-11-25T16:44:49.147Z

Note: the records below describe that earlier verification, whose documented run was `sigvision.experiment --quick --require-cpp`. A later validation of the statistical estimation extension recorded: tests (`.venv/bin/python -m unittest discover -s tests -v`) exit code 0, output SHA-256 `d3ba58091891f2bd4f0901001efac66a32ef546791359b6e081d645111daba79`; build (`make build`) exit code 0, output SHA-256 `a936cfa7597342406a4c81d8c35aa290dff939bd69bff50b6eb363aa4cf3247e`; documented run (`.venv/bin/python -m sigvision.estimation --quick`) exit code 0, output SHA-256 `c64ac3da8c87eeba10890ae65de6b6ea4fdda617251e2a7b95b43d5095355f7d`.

## tests

Command: `{"executable":".venv/bin/python","args":["-m","unittest","discover","-s","tests","-v"]}`

Exit code: 0

Output SHA-256: `3272c18bfe201a7387ac9b8d956f1ab2c56e228f4e82c54858b740aaeece6b49`

## build

Command: `{"executable":"make","args":["build"]}`

Exit code: 0

Output SHA-256: `a936cfa7597342406a4c81d8c35aa290dff939bd69bff50b6eb363aa4cf3247e`

## documented_run

Command: `{"executable":".venv/bin/python","args":["-m","sigvision.experiment","--quick","--require-cpp"]}`

Exit code: 0

Output SHA-256: `c650f4dd16a877ac6778fedbfdbcb2fcad7cb13e5f45280969e3623f498df82e`

Raw execution output remains in private application storage.
