use super::{FORMAT_CHARACTER_RANGES, is_escaped_in_report};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const UNICODE_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/unicode/DerivedGeneralCategory-15.1.0-Cf.txt"
));
const UNICODE_LICENSE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/unicode/LICENSE.txt"
));

struct CategoryFixture {
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

fn parse_category_fixture(
    text: &str,
    heading: &str,
    category: &str,
) -> Result<CategoryFixture, &'static str> {
    let mut fixture = CategoryFixture {
        ranges: Vec::new(),
        scalars: BTreeSet::new(),
    };
    let mut started = false;
    let mut finished = false;
    for line in text.lines().map(str::trim) {
        if line.starts_with("# General_Category=") || line.starts_with("# Selected_Hangul_Fillers=")
        {
            if line != heading || started {
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
            return Err("record outside category section");
        }
        let record = line.split_once('#').map_or(line, |(record, _)| record);
        let (range, record_category) =
            record.split_once(';').ok_or("missing category separator")?;
        if record_category.trim() != category {
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
        return Err("incomplete or empty category section");
    }
    Ok(fixture)
}

fn parse_cf_fixture(text: &str) -> Result<CategoryFixture, &'static str> {
    parse_category_fixture(text, "# General_Category=Format", "Cf")
}

fn parse_unicode_fixture(text: &str) -> Result<[CategoryFixture; 3], &'static str> {
    let zs_start = text
        .find("# General_Category=Space_Separator\n")
        .ok_or("missing space separator section")?;
    let (cf, rest) = text.split_at(zs_start);
    let fillers_start = rest
        .find("# Selected_Hangul_Fillers=Other_Letter\n")
        .ok_or("missing selected filler section")?;
    let (zs, fillers) = rest.split_at(fillers_start);
    Ok([
        parse_cf_fixture(cf)?,
        parse_category_fixture(zs, "# General_Category=Space_Separator", "Zs")?,
        parse_category_fixture(fillers, "# Selected_Hangul_Fillers=Other_Letter", "Lo")?,
    ])
}

#[test]
fn unicode_fixture_has_pinned_identity_and_version() {
    assert_eq!(
        UNICODE_FIXTURE.lines().next(),
        Some("# DerivedGeneralCategory-15.1.0.txt")
    );
    for (bytes, expected) in [
        (
            UNICODE_FIXTURE.as_bytes(),
            "8ff93c3bddd007a7b313f822fa40764455618f426d67e47d759d376655636a9e",
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
    let [fixture, _, _] =
        parse_unicode_fixture(UNICODE_FIXTURE).expect("valid pinned Unicode extract");
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
fn unicode_report_escape_policy_matches_pinned_15_1_fixture() {
    let [cf, zs, fillers] =
        parse_unicode_fixture(UNICODE_FIXTURE).expect("valid pinned Unicode extract");
    assert_eq!(zs.ranges.len(), 7);
    assert_eq!(zs.scalars.len(), 17);
    assert_eq!(fillers.scalars.len(), 4);
    assert!(zs.scalars.contains(&' '));
    let mut expected = cf.scalars;
    expected.extend(zs.scalars);
    assert!(expected.remove(&' '));
    expected.extend(fillers.scalars);
    expected.extend(['\\', '\u{2028}', '\u{2029}']);
    let mut actual = BTreeSet::new();
    for character in (0..=0x10FFFF).filter_map(char::from_u32) {
        if character.is_control() {
            expected.insert(character);
        }
        if is_escaped_in_report(character) {
            actual.insert(character);
        }
    }
    let missing: Vec<_> = expected.difference(&actual).collect();
    let extra: Vec<_> = actual.difference(&expected).collect();
    assert!(missing.is_empty(), "missing escaped scalars: {missing:?}");
    assert!(extra.is_empty(), "extra escaped scalars: {extra:?}");
    assert!(!is_escaped_in_report(' '));
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

#[test]
fn unicode_fixture_rejects_malformed_space_and_filler_sections() {
    for (heading, category, record) in [
        ("# General_Category=Space_Separator", "Zs", "0020"),
        ("# Selected_Hangul_Fillers=Other_Letter", "Lo", "115F"),
    ] {
        let valid = format!("{heading}\n{record} ; {category}\n# Total code points: 1\n");
        assert!(parse_category_fixture(&valid, heading, category).is_ok());
        for text in [
            valid.replace(&format!("; {category}"), "; Cf"),
            valid.replace("# Total code points: 1", "# Total code points: 2"),
            format!("{valid}{record} ; {category}\n"),
            format!("{heading}\n{valid}"),
        ] {
            assert!(
                parse_category_fixture(&text, heading, category).is_err(),
                "accepted {text:?}"
            );
        }
    }
    for text in [
        UNICODE_FIXTURE.replace("# General_Category=Space_Separator\n", ""),
        UNICODE_FIXTURE.replace("# Selected_Hangul_Fillers=Other_Letter\n", ""),
        UNICODE_FIXTURE.replace(
            "# General_Category=Space_Separator",
            "# General_Category=Format",
        ),
        UNICODE_FIXTURE.replace(
            "# Selected_Hangul_Fillers=Other_Letter",
            "# General_Category=Other_Letter",
        ),
    ] {
        assert!(parse_unicode_fixture(&text).is_err(), "accepted {text:?}");
    }
}
