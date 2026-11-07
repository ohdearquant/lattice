use super::FORMAT_CHARACTER_RANGES;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const CF_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/unicode/DerivedGeneralCategory-15.1.0-Cf.txt"
));
const UNICODE_LICENSE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/unicode/LICENSE.txt"
));

struct CfFixture {
    ranges: Vec<(char, char)>,
    scalars: BTreeSet<char>,
}

fn parse_scalar(hex: &str) -> Result<char, &'static str> {
    if !(4..=6).contains(&hex.len()) || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid scalar spelling");
    }
    let value = u32::from_str_radix(hex, 16).map_err(|_| "invalid scalar value")?;
    char::from_u32(value).ok_or("not a Unicode scalar")
}

fn parse_cf_fixture(text: &str) -> Result<CfFixture, &'static str> {
    let mut fixture = CfFixture {
        ranges: Vec::new(),
        scalars: BTreeSet::new(),
    };
    let mut started = false;
    let mut finished = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with("# General_Category=") {
            if line != "# General_Category=Format" || started {
                return Err("unexpected or repeated category header");
            }
            started = true;
            continue;
        }
        if let Some(total) = line.strip_prefix("# Total code points:") {
            let total: usize = total.trim().parse().map_err(|_| "invalid total")?;
            if !started || finished || total != fixture.scalars.len() || total == 0 {
                return Err("unexpected or inconsistent total");
            }
            finished = true;
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !started || finished {
            return Err("record outside format section");
        }
        let record = line.split_once('#').map_or(line, |(record, _)| record);
        let (range, category) = record.split_once(';').ok_or("missing category separator")?;
        if category.trim() != "Cf" {
            return Err("unexpected category or extra field");
        }
        let range = range.trim();
        let (first, last) = range.split_once("..").unwrap_or((range, range));
        let (first, last) = (parse_scalar(first)?, parse_scalar(last)?);
        if first > last || fixture.ranges.last().is_some_and(|&(_, end)| first <= end) {
            return Err("descending, overlapping or unordered range");
        }
        for value in first as u32..=last as u32 {
            let scalar = char::from_u32(value).ok_or("range crosses non-scalars")?;
            fixture.scalars.insert(scalar);
        }
        fixture.ranges.push((first, last));
    }
    if !finished || fixture.ranges.is_empty() {
        return Err("incomplete or empty format section");
    }
    Ok(fixture)
}

#[test]
fn unicode_cf_fixture_has_pinned_identity_and_version() {
    assert_eq!(
        CF_FIXTURE.lines().next(),
        Some("# DerivedGeneralCategory-15.1.0.txt")
    );
    for (bytes, expected) in [
        (
            CF_FIXTURE.as_bytes(),
            "dd770bd7b023b7e4181a5d27942f8cea5710f43c5e64923936e89f77089c6c0c",
        ),
        (
            UNICODE_LICENSE,
            "fe5c62b543e287981db198f2acfa0ca732d12591a1024536dc8fd85dacd77104",
        ),
    ] {
        let digest: String = Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(digest, expected);
    }
}

#[test]
fn unicode_cf_table_matches_pinned_15_1_fixture() {
    let fixture = parse_cf_fixture(CF_FIXTURE).expect("valid pinned Unicode Cf extract");
    assert_eq!(fixture.ranges.len(), 21);
    assert_eq!(fixture.scalars.len(), 170);
    assert!(
        FORMAT_CHARACTER_RANGES
            .windows(2)
            .all(|pair| pair[0].1 < pair[1].0)
    );
    let mut actual = BTreeSet::new();
    for &(first, last) in FORMAT_CHARACTER_RANGES {
        assert!(first <= last);
        for value in first as u32..=last as u32 {
            let scalar = char::from_u32(value).expect("production range contains only scalars");
            assert!(actual.insert(scalar), "overlapping production range");
        }
    }
    let missing: Vec<_> = fixture.scalars.difference(&actual).collect();
    let extra: Vec<_> = actual.difference(&fixture.scalars).collect();
    assert!(missing.is_empty(), "missing Cf scalars: {missing:?}");
    assert!(extra.is_empty(), "extra non-Cf scalars: {extra:?}");
    for scalars in [&fixture.scalars, &actual] {
        assert!(scalars.contains(&'\u{2064}'));
        assert!(!scalars.contains(&'\u{2065}'));
        assert!(scalars.contains(&'\u{2066}'));
    }
}

#[test]
fn unicode_cf_fixture_rejects_malformed_records() {
    for (record, total) in [
        ("00AD Cf", 1),
        ("00AD ; Cf ; Cf", 1),
        ("00AD ; Cn", 1),
        ("not-hex ; Cf", 1),
        (" ; Cf", 1),
        ("00AD.. ; Cf", 1),
        ("00AD....00AE ; Cf", 2),
        ("00AE..00AD ; Cf", 2),
        ("110000 ; Cf", 1),
        ("D800 ; Cf", 1),
        ("D7FF..E000 ; Cf", 2),
        ("00AD ; Cf\n00AD ; Cf", 2),
        ("0600 ; Cf\n00AD ; Cf", 2),
        ("0600..0602 ; Cf\n0602..0603 ; Cf", 4),
        ("00AD ; Cf", 2),
        ("", 0),
    ] {
        let text = format!("# General_Category=Format\n{record}\n# Total code points: {total}\n");
        assert!(parse_cf_fixture(&text).is_err(), "accepted {text:?}");
    }
}

#[test]
fn unicode_cf_fixture_rejects_malformed_sections() {
    let valid = "# General_Category=Format\n00AD ; Cf\n# Total code points: 1\n";
    assert_eq!(
        parse_cf_fixture(valid).unwrap().ranges,
        [('\u{AD}', '\u{AD}')]
    );
    for text in [
        String::new(),
        valid.replace("# General_Category=Format\n", ""),
        valid.replace("General_Category=Format", "General_Category=Control"),
        valid.replace("# Total code points: 1\n", ""),
        valid.replace("# Total code points: 1", "# Total code points: invalid"),
        format!("# General_Category=Format\n{valid}"),
        format!("{valid}# Total code points: 1\n"),
        format!("00AD ; Cf\n{valid}"),
        format!("{valid}0600 ; Cf\n"),
    ] {
        assert!(parse_cf_fixture(&text).is_err(), "accepted {text:?}");
    }
}
