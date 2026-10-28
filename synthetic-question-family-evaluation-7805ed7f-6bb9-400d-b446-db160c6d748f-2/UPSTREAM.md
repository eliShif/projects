Original source: https://github.com/eliShif/projects/tree/e5ae323b7c9e3689545bbcfb18439326df0e0dd6
Authorship is not inferred; this adaptation is AI-assisted.

This workspace extension reuses `portfolio/research_cases.py` and
`portfolio/research_lab.py`. The original remote was not modified. The optional
pool-aware generator mode is used only by the question-family experiment;
the existing default dev/test datasets have regression checks against their
pre-extension output fingerprints.

SHA-256 of inspected files before this extension:

| File | SHA-256 |
|---|---|
| `portfolio/research_cases.py` | `d9d398f1096beca68e956e2a1a034e90c54aeac45d4ad1c4996a57cdb72a8806` |
| `portfolio/research_lab.py` | `5a714a5bad556c4537e50dbab55924626367c045a097207ffa118daa6e013c88` |
| `tests/test_research_lab.py` | `cf6cc61e71c1394e1255c17d4d47c51bff7721248d1e20e727c2c7ccf35d8dca` |

The regenerated experiment is version 2.0.0, with current runtime source hashes
in `results/research_lab/protocol.json`. Other saved portfolio results remain
inherited artifacts and are not revalidated by this extension.
