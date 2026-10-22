/// Removes `//` comments and ordinary quoted strings while preserving source length;
/// block comments, raw strings, and character literals are outside its limits.
pub(super) fn strip_comments_and_strings(source: &str) -> String {
    let mut bytes = source.as_bytes().to_vec();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            bytes[index] = b' ';
            bytes[index + 1] = b' ';
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                bytes[index] = b' ';
                index += 1;
            }
        } else if bytes[index] == b'"' {
            bytes[index] = b' ';
            index += 1;
            while index < bytes.len() {
                match bytes[index] {
                    b'\\' => {
                        bytes[index] = b' ';
                        index += 1;
                        if index < bytes.len() && bytes[index] != b'\n' {
                            bytes[index] = b' ';
                        }
                        index += 1;
                    }
                    b'"' => {
                        bytes[index] = b' ';
                        index += 1;
                        break;
                    }
                    b'\n' => index += 1,
                    _ => {
                        bytes[index] = b' ';
                        index += 1;
                    }
                }
            }
        } else {
            index += 1;
        }
    }
    String::from_utf8(bytes).expect("source remains valid UTF-8 after scrubbing")
}
