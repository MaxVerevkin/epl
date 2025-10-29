use crate::ast;

/// Print an error in 'diagnostic' style, pointing to the location of the problem
pub fn print_error(path: &str, src: &str, error: ast::Error) {
    eprint!("error: ");
    match error.kind {
        ast::ErrorKind::Lex(error_kind) => match error_kind {
            crate::lex::ErrorKind::UnexpectedChar(ch) => {
                eprintln!("unexpected character {ch:?}");
            }
            crate::lex::ErrorKind::UnclosedDelimeter(del) => {
                eprintln!("unclosed delimiter {del:?}");
            }
            crate::lex::ErrorKind::UnknownEscapeSequence(seq) => {
                eprintln!("unknown escape sequence {seq:?}");
            }
        },
        ast::ErrorKind::UnexpectedToken { expected, got } => {
            eprint!("expected {expected}, found ");
            match got {
                Some(got) => eprintln!("{got:?}"),
                None => eprintln!("EOF"),
            }
        }
    }

    if let Some(span) = error.span {
        let line_number = src.as_bytes()[..span.start]
            .iter()
            .filter(|b| **b == b'\n')
            .count()
            + 1;
        let line_start = src.as_bytes()[..span.start]
            .iter()
            .rposition(|b| *b == b'\n')
            .map_or(0, |i| i + 1);
        let line_end = src.as_bytes()[span.end..]
            .iter()
            .position(|b| *b == b'\n')
            .map_or(src.len(), |i| span.end + i);
        eprintln!("\nat {path}:{line_number}");
        let mut next_line = String::new();
        for (i, ch) in src[line_start..line_end].char_indices() {
            if ch == '\t' {
                eprint!("    ");
            } else {
                eprint!("{ch}");
            }
            let in_span = span.contains(i + line_start);
            match ch {
                '\n' => {
                    eprintln!("{next_line}");
                    next_line.clear();
                }
                '\t' => {
                    next_line.push_str(match in_span {
                        true => "^^^^",
                        false => "    ",
                    });
                }
                _ => {
                    next_line.push(match in_span {
                        true => '^',
                        false => ' ',
                    });
                }
            }
        }
        if !next_line.is_empty() {
            eprintln!("\n{next_line}");
        }
    }
}
