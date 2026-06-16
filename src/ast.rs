use std::collections::VecDeque;
use std::fmt;

use crate::common::{ArithmeticOp, BinaryOp, CmpOp};
use crate::lex;

/// The abstract syntax tree representation of a source code file
#[derive(Debug)]
pub struct Ast {
    pub items: Vec<Item>,
}

/// A top level item
#[derive(Debug)]
pub struct Item {
    pub annotations: Vec<Annotation>,
    pub kind: ItemKind,
}

/// An annotation
#[derive(Debug)]
pub struct Annotation {
    pub at_symbol_span: lex::Span,
    pub ident: Ident,
}

impl Annotation {
    /// Get the span of this annotation
    pub fn span(&self) -> lex::Span {
        self.at_symbol_span.join(self.ident.span)
    }
}

/// A top level item kind
pub enum ItemKind {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
}

/// A function definition or declaration
#[derive(Debug)]
pub struct Function {
    pub name: Ident,
    pub args: Vec<FunctionArg>,
    pub is_variadic: bool,
    pub return_ty: Option<Type>,
    pub body: Option<BlockExpr>,
}

/// An argument in function definition or declaration
#[derive(Debug)]
pub struct FunctionArg {
    pub name: Ident,
    pub ty: Type,
}

/// A struct definition
#[derive(Debug)]
pub struct Struct {
    pub name: Ident,
    pub type_parameters: Vec<TypeParameter>,
    pub fields: Vec<StructField>,
}

/// A field of a struct definition
#[derive(Debug)]
pub struct StructField {
    pub name: Ident,
    pub ty: Type,
}

/// An enum definition
#[derive(Debug)]
#[expect(unused)]
pub struct Enum {
    pub name: Ident,
    pub type_parameters: Vec<TypeParameter>,
    pub entries: Vec<EnumEntry>,
}

/// Aa entry of an enum definition
#[derive(Debug)]
#[expect(unused)]
pub struct EnumEntry {
    pub name: Ident,
    pub ty: Option<Type>,
}

/// An identifier with its span
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ident {
    pub span: lex::Span,
    pub value: String,
}

/// A type with its span
pub enum Type {
    Never(lex::Span),
    Ident { ident: Ident, type_arguments: Option<TypeArguments> },
    Ptr { star_span: lex::Span, pointee: Box<Type> },
    Array { element_type: Box<Type>, length: Box<Expr>, span: lex::Span },
}

/// A generic type parameter
#[derive(Debug)]
pub struct TypeParameter {
    pub name: Ident,
}

#[derive(Debug)]
pub struct TypeArguments {
    pub span: lex::Span,
    pub arguments: Vec<Type>,
}

impl Type {
    /// Get the span of this expression
    pub fn span(&self) -> lex::Span {
        match self {
            Type::Never(span) => *span,
            Type::Ident { ident, type_arguments } => match type_arguments {
                Some(type_arguments) => type_arguments.span.join(ident.span),
                None => ident.span,
            },
            Type::Ptr { star_span, pointee } => star_span.join(pointee.span()),
            Type::Array { span, .. } => *span,
        }
    }
}

/// A block expression
#[derive(Debug)]
pub struct BlockExpr {
    pub statements: Vec<Statement>,
    pub final_expr: Option<Box<Expr>>,
    pub opening_brace_span: lex::Span,
    pub closing_brace_span: lex::Span,
}

impl BlockExpr {
    pub fn into_expr(self) -> Expr {
        Expr {
            span: self.span(),
            kind: ExprKind::Block(self),
        }
    }
}

/// A struct initializer field
#[derive(Debug)]
pub struct StructInitializerField {
    pub name: Ident,
    pub value: Expr,
}

/// A statement
#[derive(Debug)]
pub enum Statement {
    Let(LetStatement),
    Expr(Expr),
}

/// A statement
#[derive(Debug)]
pub enum LetStatement {
    WithValue { name: Ident, ty: Option<Type>, value: Box<Expr> },
    WithoutValue { name: Ident, ty: Type },
}

#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: lex::Span,
}

impl Expr {
    pub fn is_with_block(&self) -> bool {
        matches!(
            self.kind,
            ExprKind::Block(..)
                | ExprKind::If(..)
                | ExprKind::Loop(..)
                | ExprKind::While(..)
                | ExprKind::For(..)
                | ExprKind::StructInitializer(..)
        )
    }
}

#[derive(Debug)]
pub enum ExprKind {
    Ident(Ident),
    Block(BlockExpr),
    If(Box<Expr>, BlockExpr, Option<Box<Expr>>),
    Loop(BlockExpr),
    While(Box<Expr>, BlockExpr),
    For(Ident, Option<Type>, Box<Expr>, BlockExpr),
    ArrayInitializer(Vec<Expr>),
    StructInitializer(Vec<StructInitializerField>),
    Return(Option<Box<Expr>>),
    Break(Option<Box<Expr>>),
    Continue,
    Literal(Literal),
    FunctionCallExpr(Ident, Vec<Expr>),
    Assignment(Box<Expr>, Box<Expr>),
    CompoundAssignment(Box<Expr>, ArithmeticOp, Box<Expr>),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    AsCast(Box<Expr>, Type),
    TypeAscription(Box<Expr>, Type),
    Comptime(Box<Expr>),
    Range(Box<Expr>, Box<Expr>),
    FieldAccess(Box<Expr>, Ident),
    Index(Box<Expr>, Box<Expr>),
}

impl BlockExpr {
    /// Get the span of this expression
    pub fn span(&self) -> lex::Span {
        self.opening_brace_span.join(self.closing_brace_span)
    }
}

/// A literal expression value
pub enum Literal {
    Undefined,
    Null,
    Number(i128, Option<Ident>),
    String(String),
    Bool(bool),
}

/// A unary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
    AddressOf,
    Dereference,
}

/// A parser for the source code
pub struct Parser<'a> {
    lexer: lex::Lexer<'a>,
    lookahead: VecDeque<(lex::Span, lex::Token)>,
}

/// An error during parsing with its span
#[derive(Debug)]
pub struct Error {
    pub span: Option<lex::Span>,
    pub kind: ErrorKind,
}

/// A type of error during parsing
#[derive(Debug)]
pub enum ErrorKind {
    Lex(lex::ErrorKind),
    UnexpectedToken { expected: String, got: Option<lex::Token> },
    LetNoValueNoType,
    VariadicIsNotLast,
    DuplicateAnnotation,
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given source code
    pub fn new(src: &'a str) -> Self {
        Self {
            lexer: lex::Lexer::new(src),
            lookahead: VecDeque::new(),
        }
    }

    /// Parse into AST consuming this parser
    pub fn parse(mut self) -> Result<Ast, Error> {
        let mut items = Vec::new();
        while let Some(item) = self.next_item()? {
            items.push(item);
        }
        Ok(Ast { items })
    }
}

/// Parser implementation
impl Parser<'_> {
    /// Peek the next token without consuming it
    fn peek_token(&mut self) -> Result<Option<&lex::Token>, Error> {
        self.lookahead(0)
    }

    /// Peek the nth token without consuming any tokens
    fn lookahead(&mut self, n: usize) -> Result<Option<&lex::Token>, Error> {
        while self.lookahead.len() <= n
            && let Some(next) = self.lexer.next()
        {
            self.lookahead.push_back(next.map_err(|e| Error {
                span: Some(e.span),
                kind: ErrorKind::Lex(e.kind),
            })?);
        }
        Ok(self.lookahead.get(n).map(|(_s, t)| t))
    }

    /// Consume the next token
    fn consume_token(&mut self) -> Result<Option<(lex::Span, lex::Token)>, Error> {
        self.peek_token()?;
        Ok(self.lookahead.pop_front())
    }

    /// Expect the next token to be a given keyword and consume it
    fn expect_keyword(&mut self, keyword: lex::Keyword) -> Result<lex::Span, Error> {
        match self.consume_token()? {
            Some((span, lex::Token::Keyword(k))) if k == keyword => Ok(span),
            got => Err(Error {
                span: got.as_ref().map(|t| t.0),
                kind: ErrorKind::UnexpectedToken {
                    expected: format!("keyword {keyword:?}"),
                    got: got.map(|t| t.1),
                },
            }),
        }
    }

    /// Expect the next token to be a given punctuation and consume it
    fn expect_punct(&mut self, punct: lex::Punct) -> Result<lex::Span, Error> {
        match self.consume_token()? {
            Some((span, lex::Token::Punct(p))) if p == punct => Ok(span),
            got => Err(Error {
                span: got.as_ref().map(|t| t.0),
                kind: ErrorKind::UnexpectedToken {
                    expected: format!("punct {punct:?}"),
                    got: got.map(|t| t.1),
                },
            }),
        }
    }

    /// Create an error indicating unexpected token
    fn consume_unexpected_token<T>(&mut self, expected: impl Into<String>) -> Result<T, Error> {
        let got = self.consume_token()?;
        Err(Error {
            span: got.as_ref().map(|t| t.0),
            kind: ErrorKind::UnexpectedToken {
                expected: expected.into(),
                got: got.map(|t| t.1),
            },
        })
    }

    /// Parse identifier
    fn next_ident(&mut self) -> Result<Ident, Error> {
        match self.consume_token()? {
            Some((span, lex::Token::Ident(value))) => Ok(Ident { span, value }),
            got => Err(Error {
                span: got.as_ref().map(|t| t.0),
                kind: ErrorKind::UnexpectedToken {
                    expected: String::from("identifier"),
                    got: got.map(|t| t.1),
                },
            }),
        }
    }

    /// Parse type arguments, if they are present
    fn next_opt_type_arguments(&mut self) -> Result<Option<TypeArguments>, Error> {
        Ok(if self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::CmpL)) {
            let (left_angle_bracket_span, _) = self.consume_token()?.unwrap();
            let arguments = self.parse_delimited(lex::Punct::Comma, lex::Punct::CmpG, |parser| parser.next_type())?;
            let right_angle_bracket_span = self.expect_punct(lex::Punct::CmpG)?;
            Some(TypeArguments {
                span: left_angle_bracket_span.join(right_angle_bracket_span),
                arguments,
            })
        } else {
            None
        })
    }

    /// Parse type parameters, if they are present
    fn next_opt_type_parameters(&mut self) -> Result<Vec<TypeParameter>, Error> {
        Ok(if self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::CmpL)) {
            self.consume_token()?.unwrap();
            let parameters = self.parse_delimited(lex::Punct::Comma, lex::Punct::CmpG, |parser| {
                parser.next_type_parameter()
            })?;
            self.expect_punct(lex::Punct::CmpG)?;
            parameters
        } else {
            Vec::new()
        })
    }

    /// Parse type parameter
    pub fn next_type_parameter(&mut self) -> Result<TypeParameter, Error> {
        let name = self.next_ident()?;
        Ok(TypeParameter { name })
    }

    /// Parse type
    fn next_type(&mut self) -> Result<Type, Error> {
        match self.consume_token()? {
            Some((span, lex::Token::Ident(value))) => {
                let ident = Ident { span, value };
                let type_arguments = self.next_opt_type_arguments()?;
                Ok(Type::Ident { ident, type_arguments })
            }
            Some((span, lex::Token::Punct(lex::Punct::Exclam))) => Ok(Type::Never(span)),
            Some((star_span, lex::Token::Punct(lex::Punct::Star))) => {
                let pointee = self.next_type()?;
                Ok(Type::Ptr {
                    star_span,
                    pointee: Box::new(pointee),
                })
            }
            Some((left_bracket_span, lex::Token::Punct(lex::Punct::LeftBracket))) => {
                let element_type = self.next_type()?;
                self.expect_punct(lex::Punct::Semicolon)?;
                let length = self.next_expr()?;
                let right_bracket_span = self.expect_punct(lex::Punct::RightBracket)?;
                Ok(Type::Array {
                    element_type: Box::new(element_type),
                    length: Box::new(length),
                    span: left_bracket_span.join(right_bracket_span),
                })
            }
            got => Err(Error {
                span: got.as_ref().map(|t| t.0),
                kind: ErrorKind::UnexpectedToken {
                    expected: String::from("type"),
                    got: got.map(|t| t.1),
                },
            }),
        }
    }

    /// Parse item
    fn next_item(&mut self) -> Result<Option<Item>, Error> {
        let mut annotations = Vec::<Annotation>::new();

        while self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::At)) {
            let (at_symbol_span, _) = self.consume_token()?.unwrap();
            let ident = self.next_ident()?;
            let annotation = Annotation { at_symbol_span, ident };
            if annotations.iter().any(|a| a.ident.value == annotation.ident.value) {
                return Err(Error {
                    span: Some(annotation.span()),
                    kind: ErrorKind::DuplicateAnnotation,
                });
            }
            annotations.push(annotation);
        }

        match self.peek_token()? {
            Some(lex::Token::Keyword(lex::Keyword::Fn)) => self.next_function(annotations).map(Some),
            Some(lex::Token::Keyword(lex::Keyword::Struct)) => self.next_struct(annotations).map(Some),
            Some(lex::Token::Keyword(lex::Keyword::Enum)) => self.next_enum(annotations).map(Some),
            None => Ok(None),
            _ => self.consume_unexpected_token("an item (function or struct)"),
        }
    }

    /// Parse function
    fn next_function(&mut self, annotations: Vec<Annotation>) -> Result<Item, Error> {
        self.expect_keyword(lex::Keyword::Fn)?;
        let name = self.next_ident()?;
        self.expect_punct(lex::Punct::LeftParen)?;
        let mut args = Vec::new();
        let mut is_variadic = false;
        loop {
            match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::RightParen)) => {
                    self.consume_token()?;
                    break;
                }
                Some(lex::Token::Punct(lex::Punct::TripleDot)) => {
                    let (span, _) = self.consume_token()?.unwrap();
                    is_variadic = true;
                    if self.peek_token()? != Some(&lex::Token::Punct(lex::Punct::RightParen)) {
                        return Err(Error {
                            span: Some(span),
                            kind: ErrorKind::VariadicIsNotLast,
                        });
                    }
                    self.expect_punct(lex::Punct::RightParen)?;
                    break;
                }
                Some(lex::Token::Ident(_)) => {
                    let name = self.next_ident()?;
                    self.expect_punct(lex::Punct::Colon)?;
                    let ty = self.next_type()?;
                    args.push(FunctionArg { name, ty });
                    match self.peek_token()? {
                        Some(lex::Token::Punct(lex::Punct::Comma)) => {
                            self.consume_token()?;
                        }
                        Some(lex::Token::Punct(lex::Punct::RightParen)) => {
                            self.consume_token()?;
                            break;
                        }
                        _ => {
                            return self.consume_unexpected_token("function argument, ',' or ')'");
                        }
                    }
                }
                _ => {
                    return self.consume_unexpected_token("function argument or ')'");
                }
            }
        }
        let return_ty = match self.peek_token()? {
            Some(lex::Token::Punct(lex::Punct::Arrow)) => {
                self.consume_token()?;
                Some(self.next_type()?)
            }
            _ => None,
        };
        let body = match self.peek_token()? {
            Some(lex::Token::Punct(lex::Punct::LeftBrace)) => Some(self.next_block_expr()?),
            Some(lex::Token::Punct(lex::Punct::Semicolon)) => {
                self.consume_token()?;
                None
            }
            _ => {
                if return_ty.is_none() {
                    return self.consume_unexpected_token("function return type, function body or ';'");
                } else {
                    return self.consume_unexpected_token("function body or ';'");
                }
            }
        };
        Ok(Item {
            annotations,
            kind: ItemKind::Function(Function {
                name,
                args,
                is_variadic,
                return_ty,
                body,
            }),
        })
    }

    /// Parse struct
    fn next_struct(&mut self, annotations: Vec<Annotation>) -> Result<Item, Error> {
        self.expect_keyword(lex::Keyword::Struct)?;
        let name = self.next_ident()?;
        let type_parameters = self.next_opt_type_parameters()?;
        self.expect_punct(lex::Punct::LeftBrace)?;
        let fields = self.parse_delimited(lex::Punct::Comma, lex::Punct::RightBrace, |parser| {
            let name = parser.next_ident()?;
            parser.expect_punct(lex::Punct::Colon)?;
            let ty = parser.next_type()?;
            Ok(StructField { name, ty })
        })?;
        self.expect_punct(lex::Punct::RightBrace)?;
        Ok(Item {
            annotations,
            kind: ItemKind::Struct(Struct {
                name,
                fields,
                type_parameters,
            }),
        })
    }

    /// Parse enum
    fn next_enum(&mut self, annotations: Vec<Annotation>) -> Result<Item, Error> {
        self.expect_keyword(lex::Keyword::Enum)?;
        let name = self.next_ident()?;
        let type_parameters = self.next_opt_type_parameters()?;
        self.expect_punct(lex::Punct::LeftBrace)?;
        let entries = self.parse_delimited(lex::Punct::Comma, lex::Punct::RightBrace, |parser| {
            let entry_name = parser.next_ident()?;
            let ty = match parser.peek_token()? {
                Some(&lex::Token::Punct(lex::Punct::LeftParen)) => {
                    parser.consume_token()?;
                    let ty = parser.next_type()?;
                    parser.expect_punct(lex::Punct::RightParen)?;
                    Some(ty)
                }
                _ => None,
            };
            Ok(EnumEntry { name: entry_name, ty })
        })?;
        self.expect_punct(lex::Punct::RightBrace)?;
        Ok(Item {
            annotations,
            kind: ItemKind::Enum(Enum {
                name,
                entries,
                type_parameters,
            }),
        })
    }

    /// Parse block expression
    fn next_block_expr(&mut self) -> Result<BlockExpr, Error> {
        let opening_brace_span = self.expect_punct(lex::Punct::LeftBrace)?;
        let mut statements = Vec::new();
        let mut final_expr = None;
        loop {
            match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::RightBrace)) => {
                    break;
                }
                Some(lex::Token::Punct(lex::Punct::Semicolon)) => {
                    self.consume_token()?;
                }
                Some(lex::Token::Keyword(lex::Keyword::Let)) => {
                    statements.push(self.next_let_statement()?);
                }
                _ => match self.next_expr()? {
                    expr if expr.is_with_block() => match self.peek_token()? {
                        Some(lex::Token::Punct(lex::Punct::RightBrace)) => {
                            final_expr = Some(expr);
                            break;
                        }
                        _ => statements.push(Statement::Expr(expr)),
                    },
                    expr_with_no_block => match self.peek_token()? {
                        Some(lex::Token::Punct(lex::Punct::RightBrace)) => {
                            final_expr = Some(expr_with_no_block);
                            break;
                        }
                        Some(lex::Token::Punct(lex::Punct::Semicolon)) => {
                            self.consume_token()?;
                            statements.push(Statement::Expr(expr_with_no_block));
                        }
                        _ => {
                            return self.consume_unexpected_token("';' or '}'");
                        }
                    },
                },
            }
        }
        let closing_brace_span = self.expect_punct(lex::Punct::RightBrace)?;
        Ok(BlockExpr {
            statements,
            final_expr: final_expr.map(Box::new),
            opening_brace_span,
            closing_brace_span,
        })
    }

    /// Parse let statement
    fn next_let_statement(&mut self) -> Result<Statement, Error> {
        let let_keyword_span = self.expect_keyword(lex::Keyword::Let)?;
        let name = self.next_ident()?;
        let mut ty = None;
        let mut value = None;

        if self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::Colon)) {
            self.consume_token()?;
            ty = Some(self.next_type()?);
        }

        if self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::Assign)) {
            self.consume_token()?;
            value = Some(Box::new(self.next_expr()?));
        }

        self.expect_punct(lex::Punct::Semicolon)?;

        Ok(Statement::Let(match (ty, value) {
            (Some(ty), None) => LetStatement::WithoutValue { name, ty },
            (ty, Some(value)) => LetStatement::WithValue { name, ty, value },
            (None, None) => {
                return Err(Error {
                    span: Some(let_keyword_span),
                    kind: ErrorKind::LetNoValueNoType,
                });
            }
        }))
    }

    /// Parse an expression
    fn next_expr(&mut self) -> Result<Expr, Error> {
        match self.peek_token()? {
            Some(lex::Token::Keyword(lex::Keyword::Return)) => {
                let (return_keyword_span, _) = self.consume_token()?.unwrap();
                let value = match self.peek_token()? {
                    Some(lex::Token::Punct(
                        lex::Punct::Semicolon | lex::Punct::Comma | lex::Punct::RightParen | lex::Punct::RightBrace,
                    )) => None,
                    _ => Some(Box::new(self.next_expr()?)),
                };
                Ok(Expr {
                    span: value
                        .as_ref()
                        .map_or(return_keyword_span, |val| val.span.join(return_keyword_span)),
                    kind: ExprKind::Return(value),
                })
            }
            Some(lex::Token::Keyword(lex::Keyword::Comptime)) => {
                let (comptime_span, _) = self.consume_token()?.unwrap();
                let expr = self.next_expr()?;
                Ok(Expr {
                    span: comptime_span.join(expr.span),
                    kind: ExprKind::Comptime(Box::new(expr)),
                })
            }
            Some(lex::Token::Keyword(lex::Keyword::Break)) => {
                let (break_keyword_span, _) = self.consume_token()?.unwrap();
                let value = match self.peek_token()? {
                    Some(lex::Token::Punct(
                        lex::Punct::Semicolon | lex::Punct::Comma | lex::Punct::RightParen | lex::Punct::RightBrace,
                    )) => None,
                    _ => Some(Box::new(self.next_expr()?)),
                };
                Ok(Expr {
                    span: value
                        .as_ref()
                        .map_or(break_keyword_span, |val| val.span.join(break_keyword_span)),
                    kind: ExprKind::Break(value),
                })
            }
            Some(lex::Token::Keyword(lex::Keyword::Continue)) => {
                let (continue_keyword_span, _) = self.consume_token()?.unwrap();
                Ok(Expr {
                    span: continue_keyword_span,
                    kind: ExprKind::Continue,
                })
            }
            _ => self.next_assigning_expr(),
        }
    }

    fn next_assigning_expr(&mut self) -> Result<Expr, Error> {
        let expr = self.next_or_expr()?;
        match self.peek_token()? {
            Some(lex::Token::Punct(lex::Punct::Assign)) => {
                self.consume_token()?;
                let value = self.next_or_expr()?;
                Ok(Expr {
                    span: expr.span.join(value.span),
                    kind: ExprKind::Assignment(Box::new(expr), Box::new(value)),
                })
            }
            Some(lex::Token::Punct(
                op @ (lex::Punct::AddAssign
                | lex::Punct::SubAssign
                | lex::Punct::MulAssign
                | lex::Punct::DivAssign
                | lex::Punct::RemAssign),
            )) => {
                let op = match *op {
                    lex::Punct::AddAssign => ArithmeticOp::Add,
                    lex::Punct::SubAssign => ArithmeticOp::Sub,
                    lex::Punct::MulAssign => ArithmeticOp::Mul,
                    lex::Punct::DivAssign => ArithmeticOp::Div,
                    lex::Punct::RemAssign => ArithmeticOp::Rem,
                    _ => unreachable!(),
                };
                self.consume_token()?.unwrap();
                let value = self.next_or_expr()?;
                Ok(Expr {
                    span: expr.span.join(value.span),
                    kind: ExprKind::CompoundAssignment(Box::new(expr), op, Box::new(value)),
                })
            }
            _ => Ok(expr),
        }
    }

    fn next_or_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_and_expr()?;
        while self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::LogicalOr)) {
            self.consume_token()?.unwrap();
            let rhs = self.next_and_expr()?;
            expr = Expr {
                span: expr.span.join(rhs.span),
                kind: ExprKind::Binary(Box::new(expr), BinaryOp::LogicalOr, Box::new(rhs)),
            };
        }
        Ok(expr)
    }

    fn next_and_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_comp_expr()?;
        while self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::LogicalAnd)) {
            self.consume_token()?.unwrap();
            let rhs = self.next_comp_expr()?;
            expr = Expr {
                span: expr.span.join(rhs.span),
                kind: ExprKind::Binary(Box::new(expr), BinaryOp::LogicalAnd, Box::new(rhs)),
            };
        }
        Ok(expr)
    }

    fn next_comp_expr(&mut self) -> Result<Expr, Error> {
        let expr = self.next_range_expr()?;
        let op = match self.peek_token()? {
            Some(lex::Token::Punct(lex::Punct::CmpEq)) => Some(BinaryOp::Cmp(CmpOp::Equal)),
            Some(lex::Token::Punct(lex::Punct::CmpNeq)) => Some(BinaryOp::Cmp(CmpOp::NotEqual)),
            Some(lex::Token::Punct(lex::Punct::CmpLe)) => Some(BinaryOp::Cmp(CmpOp::LessOrEqual)),
            Some(lex::Token::Punct(lex::Punct::CmpGe)) => Some(BinaryOp::Cmp(CmpOp::GreaterOrEqual)),
            Some(lex::Token::Punct(lex::Punct::CmpL)) => Some(BinaryOp::Cmp(CmpOp::Less)),
            Some(lex::Token::Punct(lex::Punct::CmpG)) => Some(BinaryOp::Cmp(CmpOp::Greater)),
            _ => None,
        };
        match op {
            Some(op) => {
                self.consume_token()?.unwrap();
                let rhs = self.next_range_expr()?;
                Ok(Expr {
                    span: expr.span.join(rhs.span),
                    kind: ExprKind::Binary(Box::new(expr), op, Box::new(rhs)),
                })
            }
            None => Ok(expr),
        }
    }

    fn next_range_expr(&mut self) -> Result<Expr, Error> {
        let expr = self.next_additive_expr()?;
        if self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::DoubleDot)) {
            self.consume_token()?.unwrap();
            let to = self.next_additive_expr()?;
            Ok(Expr {
                span: expr.span.join(to.span),
                kind: ExprKind::Range(Box::new(expr), Box::new(to)),
            })
        } else {
            Ok(expr)
        }
    }

    fn next_additive_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_multiplicative_expr()?;
        loop {
            let op = match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::Plus)) => BinaryOp::Arithmetic(ArithmeticOp::Add),
                Some(lex::Token::Punct(lex::Punct::Minus)) => BinaryOp::Arithmetic(ArithmeticOp::Sub),
                _ => break,
            };
            self.consume_token()?.unwrap();
            let rhs = self.next_multiplicative_expr()?;
            expr = Expr {
                span: expr.span.join(rhs.span),
                kind: ExprKind::Binary(Box::new(expr), op, Box::new(rhs)),
            };
        }
        Ok(expr)
    }

    fn next_multiplicative_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_as_expr()?;
        loop {
            let op = match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::Star)) => BinaryOp::Arithmetic(ArithmeticOp::Mul),
                Some(lex::Token::Punct(lex::Punct::Slash)) => BinaryOp::Arithmetic(ArithmeticOp::Div),
                Some(lex::Token::Punct(lex::Punct::Percent)) => BinaryOp::Arithmetic(ArithmeticOp::Rem),
                _ => break,
            };
            self.consume_token()?.unwrap();
            let rhs = self.next_multiplicative_expr()?;
            expr = Expr {
                span: expr.span.join(rhs.span),
                kind: ExprKind::Binary(Box::new(expr), op, Box::new(rhs)),
            };
        }
        Ok(expr)
    }

    fn next_as_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_unary_expr()?;
        while let Some(lex::Token::Keyword(kw @ (lex::Keyword::As | lex::Keyword::Of))) = self.peek_token()? {
            let is_as = *kw == lex::Keyword::As;
            self.consume_token()?.unwrap();
            let ty = self.next_type()?;
            expr = Expr {
                span: expr.span.join(ty.span()),
                kind: if is_as {
                    ExprKind::AsCast(Box::new(expr), ty)
                } else {
                    ExprKind::TypeAscription(Box::new(expr), ty)
                },
            };
        }
        Ok(expr)
    }

    fn next_unary_expr(&mut self) -> Result<Expr, Error> {
        match self.peek_token()? {
            Some(lex::Token::Punct(lex::Punct::Minus)) => {
                let (op_span, _) = self.consume_token()?.unwrap();
                Ok(match self.next_unary_expr()? {
                    Expr {
                        span,
                        kind: ExprKind::Literal(Literal::Number(number, suffix)),
                    } => Expr {
                        span: span.join(op_span),
                        kind: ExprKind::Literal(Literal::Number(-number, suffix)),
                    },
                    rhs => Expr {
                        span: op_span.join(rhs.span),
                        kind: ExprKind::Unary(UnaryOp::Negate, Box::new(rhs)),
                    },
                })
            }
            Some(lex::Token::Punct(lex::Punct::Exclam)) => {
                let (op_span, _) = self.consume_token()?.unwrap();
                let rhs = self.next_unary_expr()?;
                Ok(Expr {
                    span: op_span.join(rhs.span),
                    kind: ExprKind::Unary(UnaryOp::Not, Box::new(rhs)),
                })
            }
            Some(lex::Token::Punct(lex::Punct::Ampersand)) => {
                let (op_span, _) = self.consume_token()?.unwrap();
                let rhs = self.next_unary_expr()?;
                Ok(Expr {
                    span: op_span.join(rhs.span),
                    kind: ExprKind::Unary(UnaryOp::AddressOf, Box::new(rhs)),
                })
            }
            _ => self.next_field_access_expr(),
        }
    }

    /// Parse a field-access expr
    fn next_field_access_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_base_expr()?;
        loop {
            match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::Dot)) => {
                    self.consume_token()?.unwrap();
                    match self.peek_token()? {
                        Some(lex::Token::Ident(_)) => {
                            let name = self.next_ident()?;
                            expr = Expr {
                                span: expr.span.join(name.span),
                                kind: ExprKind::FieldAccess(Box::new(expr), name),
                            };
                        }
                        Some(lex::Token::Punct(lex::Punct::Star)) => {
                            let (star_span, _) = self.consume_token()?.unwrap();
                            expr = Expr {
                                span: star_span.join(expr.span),
                                kind: ExprKind::Unary(UnaryOp::Dereference, Box::new(expr)),
                            };
                        }
                        _ => return self.consume_unexpected_token("ident or '*'"),
                    }
                }
                Some(lex::Token::Punct(lex::Punct::LeftBracket)) => {
                    self.consume_token()?.unwrap();
                    let index = self.next_expr()?;
                    let right_bracket_span = self.expect_punct(lex::Punct::RightBracket)?;
                    expr = Expr {
                        span: expr.span.join(right_bracket_span),
                        kind: ExprKind::Index(Box::new(expr), Box::new(index)),
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    /// Parse a base expression
    fn next_base_expr(&mut self) -> Result<Expr, Error> {
        match self.peek_token()? {
            Some(lex::Token::Ident(_)) => {
                if self.lookahead(1)? == Some(&lex::Token::Punct(lex::Punct::LeftParen)) {
                    self.next_function_call_expr()
                } else {
                    self.next_ident().map(|ident| Expr {
                        span: ident.span,
                        kind: ExprKind::Ident(ident),
                    })
                }
            }
            Some(lex::Token::Literal(_)) => {
                let Some((span, lex::Token::Literal(lit))) = self.consume_token()? else {
                    unreachable!()
                };
                Ok(Expr {
                    span,
                    kind: ExprKind::Literal(match lit {
                        lex::Literal::Number(num, suffix) => {
                            Literal::Number(num, suffix.map(|s| Ident { span: s.1, value: s.0 }))
                        }
                        lex::Literal::String(str) => Literal::String(str),
                    }),
                })
            }
            Some(lex::Token::Keyword(lex::Keyword::True)) => {
                let (span, _) = self.consume_token()?.unwrap();
                Ok(Expr {
                    span,
                    kind: ExprKind::Literal(Literal::Bool(true)),
                })
            }
            Some(lex::Token::Keyword(lex::Keyword::False)) => {
                let (span, _) = self.consume_token()?.unwrap();
                Ok(Expr {
                    span,
                    kind: ExprKind::Literal(Literal::Bool(false)),
                })
            }
            Some(lex::Token::Keyword(lex::Keyword::Undefined)) => {
                let (span, _) = self.consume_token()?.unwrap();
                Ok(Expr {
                    span,
                    kind: ExprKind::Literal(Literal::Undefined),
                })
            }
            Some(lex::Token::Keyword(lex::Keyword::Null)) => {
                let (span, _) = self.consume_token()?.unwrap();
                Ok(Expr {
                    span,
                    kind: ExprKind::Literal(Literal::Null),
                })
            }
            Some(lex::Token::Punct(lex::Punct::LeftParen)) => {
                let (start_span, _) = self.consume_token()?.unwrap();
                let expr = self.next_expr()?;
                let end_span = self.expect_punct(lex::Punct::RightParen)?;
                Ok(Expr {
                    kind: expr.kind,
                    span: start_span.join(end_span),
                })
            }
            Some(lex::Token::Punct(lex::Punct::LeftBracket)) => self.next_array_initializer_expr(),
            Some(lex::Token::Punct(lex::Punct::DotLeftBrace)) => self.next_struct_initializer_expr(),
            Some(lex::Token::Punct(lex::Punct::LeftBrace)) => self.next_block_expr().map(|block| block.into_expr()),
            Some(lex::Token::Keyword(lex::Keyword::If)) => self.next_if_expr(),
            Some(lex::Token::Keyword(lex::Keyword::Loop)) => self.next_loop_expr(),
            Some(lex::Token::Keyword(lex::Keyword::While)) => self.next_while_expr(),
            Some(lex::Token::Keyword(lex::Keyword::For)) => self.next_for_expr(),
            _ => self.consume_unexpected_token("expression"),
        }
    }

    /// Parse function call
    fn next_function_call_expr(&mut self) -> Result<Expr, Error> {
        let name = self.next_ident()?;
        self.expect_punct(lex::Punct::LeftParen)?;
        let args = self.parse_delimited(lex::Punct::Comma, lex::Punct::RightParen, |parser| parser.next_expr())?;
        let right_paren_span = self.expect_punct(lex::Punct::RightParen)?;
        Ok(Expr {
            span: name.span.join(right_paren_span),
            kind: ExprKind::FunctionCallExpr(name, args),
        })
    }

    /// Parse array initializer
    fn next_array_initializer_expr(&mut self) -> Result<Expr, Error> {
        let opening_bracket_span = self.expect_punct(lex::Punct::LeftBracket)?;
        let elements =
            self.parse_delimited(lex::Punct::Comma, lex::Punct::RightBracket, |parser| parser.next_expr())?;
        let closing_bracket_span = self.expect_punct(lex::Punct::RightBracket)?;
        Ok(Expr {
            span: opening_bracket_span.join(closing_bracket_span),
            kind: ExprKind::ArrayInitializer(elements),
        })
    }

    /// Parse struct initializer
    fn next_struct_initializer_expr(&mut self) -> Result<Expr, Error> {
        let opening_brace_span = self.expect_punct(lex::Punct::DotLeftBrace)?;
        let fields = self.parse_delimited(lex::Punct::Comma, lex::Punct::RightBrace, |parser| {
            let name = parser.next_ident()?;
            parser.expect_punct(lex::Punct::Colon)?;
            let value = parser.next_expr()?;
            Ok(StructInitializerField { name, value })
        })?;
        let closing_brace_span = self.expect_punct(lex::Punct::RightBrace)?;
        Ok(Expr {
            span: opening_brace_span.join(closing_brace_span),
            kind: ExprKind::StructInitializer(fields),
        })
    }

    /// Parse if expression
    fn next_if_expr(&mut self) -> Result<Expr, Error> {
        let if_keyword_span = self.expect_keyword(lex::Keyword::If)?;
        let cond = self.next_expr()?;
        let if_true = self.next_block_expr()?;
        let if_false = if self.peek_token()? == Some(&lex::Token::Keyword(lex::Keyword::Else)) {
            self.consume_token()?;
            Some(Box::new(match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::LeftBrace)) => self.next_block_expr()?.into_expr(),
                Some(lex::Token::Keyword(lex::Keyword::If)) => self.next_if_expr()?,
                _ => return self.consume_unexpected_token("'{' or 'if'"),
            }))
        } else {
            None
        };
        Ok(Expr {
            span: if_keyword_span.join(if_false.as_ref().map_or(if_true.span(), |if_false| if_false.span)),
            kind: ExprKind::If(Box::new(cond), if_true, if_false),
        })
    }

    /// Parse loop expression
    fn next_loop_expr(&mut self) -> Result<Expr, Error> {
        let loop_keyword_span = self.expect_keyword(lex::Keyword::Loop)?;
        let body = self.next_block_expr()?;
        Ok(Expr {
            span: loop_keyword_span.join(body.span()),
            kind: ExprKind::Loop(body),
        })
    }

    /// Parse while expression
    fn next_while_expr(&mut self) -> Result<Expr, Error> {
        let while_keyword_span = self.expect_keyword(lex::Keyword::While)?;
        let cond = self.next_expr()?;
        let body = self.next_block_expr()?;
        Ok(Expr {
            span: while_keyword_span.join(body.span()),
            kind: ExprKind::While(Box::new(cond), body),
        })
    }

    /// Parse for expression
    fn next_for_expr(&mut self) -> Result<Expr, Error> {
        let for_keyword_span = self.expect_keyword(lex::Keyword::For)?;
        let i = self.next_ident()?;
        let i_ty = if self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::Colon)) {
            self.consume_token()?.unwrap();
            Some(self.next_type()?)
        } else {
            None
        };
        self.expect_keyword(lex::Keyword::In)?;
        let iterator = self.next_expr()?;
        let body = self.next_block_expr()?;
        Ok(Expr {
            span: for_keyword_span.join(body.closing_brace_span),
            kind: ExprKind::For(i, i_ty, Box::new(iterator), body),
        })
    }

    fn parse_delimited<T>(
        &mut self,
        delim: lex::Punct,
        until: lex::Punct,
        mut parse: impl FnMut(&mut Self) -> Result<T, Error>,
    ) -> Result<Vec<T>, Error> {
        let mut list = Vec::new();
        while self.peek_token()? != Some(&lex::Token::Punct(until)) {
            list.push(parse(self)?);
            match self.peek_token()? {
                Some(lex::Token::Punct(p)) if *p == delim => {
                    self.consume_token()?;
                }
                Some(lex::Token::Punct(p)) if *p == until => {
                    break;
                }
                _ => {
                    return self.consume_unexpected_token(format!("{delim:?} or {until:?}"));
                }
            }
        }
        Ok(list)
    }
}

impl fmt::Debug for ItemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function(function) => {
                f.write_str("Item::")?;
                function.fmt(f)
            }
            Self::Struct(s) => {
                f.write_str("Item::")?;
                s.fmt(f)
            }
            Self::Enum(e) => {
                f.write_str("Item::")?;
                e.fmt(f)
            }
        }
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}..{}", self.value, self.span.start, self.span.end)
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Never(_) => f.write_str("!"),
            Self::Ident { ident, type_arguments } => write!(f, "{ident:?}<{type_arguments:?}>"),
            Self::Ptr { pointee, .. } => write!(f, "ptr({pointee:?})"),
            Self::Array {
                element_type, length, ..
            } => {
                write!(f, "[{element_type:?}; {length:?}]")
            }
        }
    }
}

impl fmt::Debug for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undefined => f.write_str("undefined"),
            Self::Null => f.write_str("null"),
            Self::Number(num, _suffix) => num.fmt(f),
            Self::String(s) => s.fmt(f),
            Self::Bool(b) => b.fmt(f),
        }
    }
}
