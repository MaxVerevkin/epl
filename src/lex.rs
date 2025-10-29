/// The lexer
pub struct Lexer<'a> {
    offset: usize,
    src: &'a str,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer to lex the given source code
    pub fn new(src: &'a str) -> Self {
        Self { offset: 0, src }
    }
}

/// A span represents a slice of the input source code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// A single token
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Keyword(Keyword),
    Ident(String),
    Literal(Literal),
    Punct(Punct),
}

fn is_valid_ident_start(ch: char) -> bool {
    matches!(ch, 'a'..='z' | 'A'..='Z' | '_')
}

fn is_valid_ident_char(ch: char) -> bool {
    matches!(ch, 'a'..='z' | 'A'..='Z' | '_' | '0'..='9')
}

/// A keyword token
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Fn,
    If,
    Else,
    Loop,
    Let,
}

/// A mapping from strings to keyword tokens
const KEYWORD_MAP: &[(&str, Keyword)] = &[
    ("fn", Keyword::Fn),
    ("if", Keyword::If),
    ("else", Keyword::Else),
    ("loop", Keyword::Loop),
    ("let", Keyword::Let),
];

/// A literal token
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Number(i64),
    String(String),
}

/// A punctuation token
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Punct {
    /// ==
    CmpEq,
    /// !=
    CmpNeq,
    /// <=
    CmpLe,
    /// >=
    CmpGe,
    /// ->
    Arrow,
    /// ;
    Semicolon,
    /// :
    Colon,
    /// ,
    Comma,
    /// .
    Dot,
    /// (
    LeftParen,
    /// )
    RightParen,
    /// {
    LeftBrace,
    /// }
    RightBrace,
    /// -
    Minus,
    /// +
    Plus,
    /// <
    CmpL,
    /// >
    CmpG,
    /// *
    Star,
    /// /
    Slash,
}

/// A mapping from strings to punct tokens
const PUNCT_MAP: &[(&str, Punct)] = &[
    ("==", Punct::CmpEq),
    ("!=", Punct::CmpNeq),
    ("<=", Punct::CmpLe),
    (">=", Punct::CmpGe),
    ("->", Punct::Arrow),
    (";", Punct::Semicolon),
    (":", Punct::Colon),
    (",", Punct::Comma),
    (".", Punct::Dot),
    ("(", Punct::LeftParen),
    (")", Punct::RightParen),
    ("{", Punct::LeftBrace),
    ("}", Punct::RightBrace),
    ("-", Punct::Minus),
    ("+", Punct::Plus),
    ("<", Punct::CmpL),
    (">", Punct::CmpG),
    ("*", Punct::Star),
    ("/", Punct::Slash),
];

/// An error during lexing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub span: Span,
    pub kind: ErrorKind,
}

/// The kind of error during lexing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    UnexpectedChar(char),
    UnclosedDelimeter(char),
    UnknownEscapeSequence(String),
}

impl Iterator for Lexer<'_> {
    type Item = Result<(Span, Token), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let ch = self.peek_char()?;

            if ch.is_ascii_whitespace() {
                self.consume_char();
                continue;
            }

            if is_valid_ident_start(ch) {
                return Some(self.next_ident_or_keyword());
            }

            if ch.is_ascii_digit() {
                return Some(self.next_number_literal());
            }

            if ch == '"' {
                return Some(self.next_string_literal());
            }

            for &(punct_str, punct_token) in PUNCT_MAP {
                if self.src[self.offset..].starts_with(punct_str) {
                    let start = self.offset;
                    self.offset += punct_str.len();
                    return Some(Ok((
                        Span {
                            start,
                            end: self.offset,
                        },
                        Token::Punct(punct_token),
                    )));
                }
            }

            return Some(Err(Error {
                span: Span {
                    start: self.offset,
                    end: self.offset + ch.len_utf8(),
                },
                kind: ErrorKind::UnexpectedChar(ch),
            }));
        }
    }
}

impl Lexer<'_> {
    /// Peek the next character without consuming it
    fn peek_char(&self) -> Option<char> {
        self.src[self.offset..].chars().next()
    }

    /// Consume a single character and return its span
    ///
    /// # Panics
    ///
    /// Panics at EOF.
    fn consume_char(&mut self) -> Span {
        let len = self
            .peek_char()
            .expect("consume_char called at EOF")
            .len_utf8();
        let start = self.offset;
        self.offset += len;
        Span {
            start,
            end: self.offset,
        }
    }

    /// Get the next ident or keyword token
    ///
    /// # Panics
    ///
    /// Panics if the next character is not a valid ident start.
    fn next_ident_or_keyword(&mut self) -> Result<(Span, Token), Error> {
        let ch = self
            .peek_char()
            .expect("next_ident_or_keyword called on EOF");
        assert!(is_valid_ident_start(ch));
        let mut span = self.consume_char();
        let mut ident = ch.to_string();
        while let Some(ch) = self.peek_char()
            && is_valid_ident_char(ch)
        {
            span = self.consume_char().join(span);
            ident.push(ch);
        }
        for &(keyword_str, keyword_token) in KEYWORD_MAP {
            if keyword_str == ident {
                return Ok((span, Token::Keyword(keyword_token)));
            }
        }
        Ok((span, Token::Ident(ident)))
    }

    /// Get the next string literal token
    ///
    /// # Panics
    ///
    /// Panics if the next character is not a quote.
    fn next_string_literal(&mut self) -> Result<(Span, Token), Error> {
        assert_eq!(self.peek_char(), Some('"'));
        let mut span = self.consume_char();
        let delim_span = span;
        let mut string = String::new();
        let mut escaping = false;
        loop {
            let Some(ch) = self.peek_char() else {
                return Err(Error {
                    span: delim_span,
                    kind: ErrorKind::UnclosedDelimeter('"'),
                });
            };

            let ch_span = self.consume_char();
            span = ch_span.join(span);

            if escaping {
                escaping = false;
                match ch {
                    'n' => string.push('\n'),
                    'r' => string.push('\r'),
                    't' => string.push('\t'),
                    '\\' => string.push('\\'),
                    _ => {
                        return Err(Error {
                            span: ch_span,
                            kind: ErrorKind::UnknownEscapeSequence(ch.to_string()),
                        });
                    }
                }
            } else {
                match ch {
                    '"' => break,
                    '\\' => escaping = true,
                    _ => string.push(ch),
                }
            }
        }
        Ok((span, Token::Literal(Literal::String(string))))
    }

    /// Get the next number literal token
    ///
    /// # Panics
    ///
    /// Panics if the next character is not an ASCII digit.
    fn next_number_literal(&mut self) -> Result<(Span, Token), Error> {
        let ch = self.peek_char().expect("next_number_literal called at EOF");
        assert!(ch.is_ascii_digit());
        let mut span = self.consume_char();
        let mut number = (ch as u8 - b'0') as i64;
        while let Some(ch) = self.peek_char()
            && ch.is_ascii_digit()
        {
            span = self.consume_char().join(span);
            number = number * 10 + (ch as u8 - b'0') as i64;
        }
        Ok((span, Token::Literal(Literal::Number(number))))
    }
}

impl Span {
    /// Create a new span covering both spans and everything in between
    fn join(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn contains(self, index: usize) -> bool {
        (self.start..self.end).contains(&index)
    }
}
