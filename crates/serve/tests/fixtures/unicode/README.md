# Unicode 15.1.0 Cf fixture

`DerivedGeneralCategory-15.1.0-Cf.txt` is a **42-line extract**, not the full
Unicode Character Database file. It retains the upstream header and complete
General_Category=Format section, with their original bytes and line endings.
The doctor test parses its records independently of the Rust table, checks both
directions of scalar coverage and the U+2065 gap, and separately pins the fixture
version and bytes. `LICENSE.txt` is the accompanying Unicode license, unchanged.

## Source and identity

The full source was retrieved from the Unicode-owned `unicode-org/unicodetools`
mirror, observed at its mutable `main` ref. The decoded content is pinned by
the Git blob ID and SHA-256 below. The enclosing commit was not resolved. The
original Unicode release endpoint was not successfully fetched, so byte
equivalence between that endpoint and this mirror content was not established.
The license was retrieved separately; a shared historical revision with the
data was not established.

- Original release URL:
  <https://www.unicode.org/Public/15.1.0/ucd/extracted/DerivedGeneralCategory.txt>
- Observed mirror URL:
  <https://github.com/unicode-org/unicodetools/blob/main/unicodetools/data/ucd/15.1.0/extracted/DerivedGeneralCategory.txt>
- Full-source Git blob: `285ffa8fb83ace75fd07990be59dd99633010d48`.
- Full-source bytes/lines: 268,498 / 4,233.
- Full-source SHA-256:
  `760720ac034f96b630a3055879a744e0907184e8aa811e89ba34583a7a487e85`.
- Extract bytes/lines: 1,912 / 42.
- Extract SHA-256:
  `dd770bd7b023b7e4181a5d27942f8cea5710f43c5e64923936e89f77089c6c0c`.
- License observed URL:
  <https://github.com/unicode-org/unicodetools/blob/main/LICENSE>
- License Git blob: `d7e7973c2fd6f2586a8999a69dc21e39af26be0f`.
- License bytes/lines: 2,033 / 41.
- License SHA-256:
  `fe5c62b543e287981db198f2acfa0ca732d12591a1024536dc8fd85dacd77104`.

## Reproduce from the pinned full source

Select original lines **1–13**, followed by **3421–3449**, using one-based,
inclusive line numbers. The latter span includes both section separators, the
format heading, all 21 Cf records and the 170-code-point total. No line is
rewritten. Given the full source as the first argument and an output path as
the second, this offline Python snippet reproduces the fixture:

```python
import hashlib
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == (
    "760720ac034f96b630a3055879a744e0907184e8aa811e89ba34583a7a487e85"
)
lines = source.splitlines(keepends=True)
extract = b"".join(lines[:13] + lines[3420:3449])
assert hashlib.sha256(extract).hexdigest() == (
    "dd770bd7b023b7e4181a5d27942f8cea5710f43c5e64923936e89f77089c6c0c"
)
Path(sys.argv[2]).write_bytes(extract)
```

The source hash is checked before applying the fixed line selection. Tests do
not download data. To update this fixture, obtain and pin the replacement
source and license, derive the extract independently of the Rust table, compare
the full sets, and review any actual escaping-policy change separately.
