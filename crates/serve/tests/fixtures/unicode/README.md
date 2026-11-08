# Unicode 15.1.0 report-escaping fixture

`DerivedGeneralCategory-15.1.0-Cf.txt` retains its original filename but now
contains the complete Cf and Zs sections plus a selected Hangul-filler Lo
projection. It is **not** the complete Unicode Character Database file. The
original 42-line Cf extract is an unchanged prefix. The Unicode header, Cf and
Zs sections retain their upstream bytes and line endings; the selected Lo
records are generated, not verbatim upstream records. `LICENSE.txt` is unchanged.

The doctor tests independently parse the fixture records, check exact Cf-table
coverage and the U+2065 gap, and compare the complete report-escaping set over
all Unicode scalars. That set includes Cf, Zs except U+0020, the four selected
fillers, controls, backslash, U+2028 and U+2029. Both missing and extra production
characters fail. Ordinary ASCII space remains printable. The fixture and
license bytes are pinned separately; tests never download data.

## Source and identity

The retained DerivedGeneralCategory source came from the Unicode-owned
`unicode-org/unicodetools` mirror at its mutable `main` ref. This update reuses
those exact bytes, verified against the existing hash. The UnicodeData names
source was retrieved from the same repository by the exact Git blob below,
after resolving that file at `main`. The decoded bytes are pinned by Git blob
ID and SHA-256; enclosing commits were not resolved. Neither source's byte
equivalence with the original Unicode release endpoint was established. The
license was retrieved separately; a shared historical revision with the data
was not established.

- Original category release URL:
  <https://www.unicode.org/Public/15.1.0/ucd/extracted/DerivedGeneralCategory.txt>
- Observed category mirror URL:
  <https://github.com/unicode-org/unicodetools/blob/main/unicodetools/data/ucd/15.1.0/extracted/DerivedGeneralCategory.txt>
- Full category source Git blob: `285ffa8fb83ace75fd07990be59dd99633010d48`.
- Full category source bytes/lines: 268,498 / 4,233.
- Full category source SHA-256:
  `760720ac034f96b630a3055879a744e0907184e8aa811e89ba34583a7a487e85`.
- Original names release URL:
  <https://www.unicode.org/Public/15.1.0/ucd/UnicodeData.txt>
- Observed names mirror URL:
  <https://github.com/unicode-org/unicodetools/blob/main/unicodetools/data/ucd/15.1.0/UnicodeData.txt>
- Full names source Git blob: `bdcc41850d7f01c90902491f2aea6a9b712bb792`.
- Full names source bytes/lines: 1,914,200 / 34,931.
- Full names source SHA-256:
  `2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86`.
- Fixture bytes/lines: 2,658 / 68.
- Fixture SHA-256:
  `8ff93c3bddd007a7b313f822fa40764455618f426d67e47d759d376655636a9e`.
- License observed URL:
  <https://github.com/unicode-org/unicodetools/blob/main/LICENSE>
- License Git blob: `d7e7973c2fd6f2586a8999a69dc21e39af26be0f`.
- License bytes/lines: 2,033 / 41.
- License SHA-256:
  `fe5c62b543e287981db198f2acfa0ca732d12591a1024536dc8fd85dacd77104`.

## Selected Hangul fillers

General_Category=Lo alone does **not** select Hangul fillers. The projection is
a bounded report policy selecting the four official names below from UnicodeData
15.1.0, then checking their membership in Lo in the pinned category source.
It is neither all Lo characters nor all characters whose names contain
"FILLER". No production Rust table is an input to this selection.

| Code point | Official UnicodeData name | Names source line | Category source line and containing Lo range |
| ---------- | ------------------------- | ----------------- | -------------------------------------------- |
| U+115F     | HANGUL CHOSEONG FILLER    | 3912              | 2286: `1100..1248`                           |
| U+1160     | HANGUL JUNGSEONG FILLER   | 3913              | 2286: `1100..1248`                           |
| U+3164     | HANGUL FILLER             | 11585             | 2358: `3131..318E`                           |
| U+FFA0     | HALFWIDTH HANGUL FILLER   | 16827             | 2430: `FFA0..FFBE`                           |

The custom `Selected_Hangul_Fillers=Other_Letter` heading and four singleton
records explicitly distinguish this projection from a complete category section.
The names corroborate the selected code points; they do not derive the report
policy from category membership.

## Reproduce from the pinned full sources

Select original category lines **1–13** then **3421–3449** (the existing header
and Cf section), add a newline, then **3382–3396** (the complete Zs section).
Line numbers are one-based and inclusive. The Cf section has 21 ranges / 170
scalars; Zs has 7 ranges / 17 scalars, including U+0020. Append the explicit
four-point projection using the selected official names and Lo membership.

Given the category source, names source and output path as three arguments,
this offline Python snippet reproduces the fixture without reading Rust source:

```python
"""Reproduce the doctor fixture from pinned official Unicode 15.1.0 data."""
import hashlib
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_bytes()
names_source = Path(sys.argv[2]).read_bytes()
assert hashlib.sha256(source).hexdigest() == (
    "760720ac034f96b630a3055879a744e0907184e8aa811e89ba34583a7a487e85"
)
assert hashlib.sha256(names_source).hexdigest() == (
    "2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86"
)
selected_names = {
    "HANGUL CHOSEONG FILLER",
    "HANGUL JUNGSEONG FILLER",
    "HANGUL FILLER",
    "HALFWIDTH HANGUL FILLER",
}
selected = []
for line in names_source.decode("utf-8").splitlines():
    codepoint, name, category, *_ = line.split(";")
    if name in selected_names:
        assert category == "Lo"
        selected.append((int(codepoint, 16), name))
assert len(selected) == len(selected_names)
lo_ranges = []
for line in source.decode("utf-8").splitlines():
    record = line.split("#", 1)[0].strip()
    if not record:
        continue
    interval, category = map(str.strip, record.split(";"))
    if category == "Lo":
        bounds = interval.split("..")
        lo_ranges.append((int(bounds[0], 16), int(bounds[-1], 16)))
projection = "\n# Selected_Hangul_Fillers=Other_Letter\n"
projection += "# Selected Lo code points, not the complete category; see README.md.\n\n"
for codepoint, name in sorted(selected):
    assert any(first <= codepoint <= last for first, last in lo_ranges)
    projection += f"{codepoint:04X} ; Lo # {name}\n"
projection += "\n# Total code points: 4\n"
lines = source.splitlines(keepends=True)
extract = b"".join(lines[:13] + lines[3420:3449])
extract += b"\n" + b"".join(lines[3381:3396]) + projection.encode("utf-8")
Path(sys.argv[3]).write_bytes(extract)
print("fixture_bytes", len(extract), "fixture_lines", len(extract.splitlines()))
print("fixture_sha256", hashlib.sha256(extract).hexdigest())
```

Source hashes are checked before extraction or projection. To update the fixture,
pin replacement sources and license, derive the extract independently of the
Rust table, compare complete sets, and review an escaping-policy change separately.
