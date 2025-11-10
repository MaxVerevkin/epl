use crate::{ast, ir, lex};

pub trait SpanError {
    fn message(&self) -> String;
    fn span(&self) -> Option<lex::Span>;
}

impl SpanError for ast::Error {
    fn message(&self) -> String {
        match &self.kind {
            ast::ErrorKind::Lex(error_kind) => match error_kind {
                crate::lex::ErrorKind::UnexpectedChar(ch) => {
                    format!("unexpected character {ch:?}")
                }
                crate::lex::ErrorKind::UnclosedDelimeter(del) => {
                    format!("unclosed delimiter {del:?}")
                }
                crate::lex::ErrorKind::UnknownEscapeSequence(seq) => {
                    format!("unknown escape sequence {seq:?}")
                }
            },
            ast::ErrorKind::UnexpectedToken { expected, got } => {
                let mut s = format!("expected {expected}, found ");
                match got {
                    Some(got) => s.push_str(&format!("{got:?}")),
                    None => s.push_str("EOF"),
                }
                s
            }
            ast::ErrorKind::LetNoValueNoType => {
                String::from("let statement with no value requires type to be specified")
            }
        }
    }

    fn span(&self) -> Option<lex::Span> {
        self.span
    }
}

impl SpanError for ir::Error {
    fn message(&self) -> String {
        self.message.clone()
    }

    fn span(&self) -> Option<lex::Span> {
        self.span
    }
}

/// Print an error in 'diagnostic' style, pointing to the location of the problem
pub fn print_error(path: &str, src: &str, error: impl SpanError) {
    eprint!("error: {}", error.message());

    if let Some(span) = error.span() {
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
