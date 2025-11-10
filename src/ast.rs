use std::collections::VecDeque;
use std::fmt;

use crate::lex;

/// The abstract sytax tree representation of a source code file
#[derive(Debug, Clone)]
pub struct Ast {
    pub items: Vec<Item>,
}

/// A top level item
#[derive(Clone)]
pub enum Item {
    Function(Function),
}

/// A function definition or declaration
#[derive(Debug, Clone)]
pub struct Function {
    pub name: Ident,
    pub args: Vec<FunctionArg>,
    pub return_ty: Ident,
    pub body: Option<BlockExpr>,
}

/// An argument in function definition or declaration
#[derive(Debug, Clone)]
pub struct FunctionArg {
    pub name: Ident,
    pub ty: Ident,
}

/// An identifier with its span
#[derive(Clone)]
pub struct Ident {
    pub span: lex::Span,
    pub value: String,
}

/// A block expression
#[derive(Debug, Clone)]
pub struct BlockExpr {
    pub statements: Vec<Statement>,
    pub final_expr: Option<ExprWithNoBlock>,
    pub opening_brace_span: lex::Span,
    pub closing_brace_span: lex::Span,
}

/// An `if` expression
#[derive(Debug, Clone)]
pub struct IfExpr {
    pub cond: Box<Expr>,
    pub if_true: BlockExpr,
    pub if_false: Option<BlockExpr>,
    pub if_keyword_span: lex::Span,
}

/// A `loop` expression
#[derive(Debug, Clone)]
pub struct LoopExpr {
    pub body: BlockExpr,
    pub loop_keyword_span: lex::Span,
}

/// A statement
#[derive(Debug, Clone)]
pub enum Statement {
    Empty,
    Let(LetStatement),
    ExprWithNoBlock(ExprWithNoBlock),
    ExprWithBlock(ExprWithBlock),
}

/// A statement
#[derive(Debug, Clone)]
pub enum LetStatement {
    WithValue {
        name: Ident,
        ty: Option<Ident>,
        value: Box<Expr>,
    },
    WithoutValue {
        name: Ident,
        ty: Ident,
    },
}

/// An expression
#[derive(Debug, Clone)]
pub enum Expr {
    WithNoBlock(ExprWithNoBlock),
    WithBlock(ExprWithBlock),
}

/// An expression which cannot be a statement on its own
#[derive(Debug, Clone)]
pub enum ExprWithNoBlock {
    Return(ReturnExpr),
    Literal(LiteralExpr),
    FunctionCallExpr(FunctionCallExpr),
    Assignment(AssignmentExpr),
    Ident(Ident),
    Binary(BinaryExpr),
    Unary(UnaryExpr),
}

/// An expression which can be a statement on its own
#[derive(Debug, Clone)]
pub enum ExprWithBlock {
    Block(BlockExpr),
    If(IfExpr),
    Loop(LoopExpr),
}

impl Expr {
    /// Get the span of this expression
    pub fn span(&self) -> lex::Span {
        match self {
            Self::WithNoBlock(expr) => expr.span(),
            Self::WithBlock(expr) => expr.span(),
        }
    }
}

impl ExprWithNoBlock {
    /// Get the span of this expression
    pub fn span(&self) -> lex::Span {
        match self {
            Self::Return(return_expr) => return_expr
                .return_keyword_span
                .join(return_expr.value.span()),
            Self::Literal(literal_expr) => literal_expr.span,
            Self::FunctionCallExpr(function_call_expr) => function_call_expr
                .name
                .span
                .join(function_call_expr.args_span),
            Self::Assignment(assignment_expr) => assignment_expr
                .place
                .span()
                .join(assignment_expr.value.span()),
            Self::Ident(ident) => ident.span,
            Self::Binary(binary_expr) => binary_expr.lhs.span().join(binary_expr.rhs.span()),
            Self::Unary(unary_expr) => unary_expr.op_span.join(unary_expr.rhs.span()),
        }
    }
}

impl ExprWithBlock {
    /// Get the span of this expression
    pub fn span(&self) -> lex::Span {
        match self {
            Self::Block(block_expr) => block_expr.span(),
            Self::If(if_expr) => if_expr.if_keyword_span.join(
                if_expr
                    .if_false
                    .as_ref()
                    .map_or_else(|| if_expr.if_true.span(), |e| e.span()),
            ),
            Self::Loop(loop_expr) => loop_expr.loop_keyword_span.join(loop_expr.body.span()),
        }
    }
}

impl BlockExpr {
    /// Get the span of this expression
    pub fn span(&self) -> lex::Span {
        self.opening_brace_span.join(self.closing_brace_span)
    }
}

/// A return expression
#[derive(Debug, Clone)]
pub struct ReturnExpr {
    pub return_keyword_span: lex::Span,
    pub value: Box<Expr>,
}

/// A literal expression with its span
#[derive(Debug, Clone)]
pub struct LiteralExpr {
    pub span: lex::Span,
    pub value: LiteralExprValue,
}

/// A literal expression value
#[derive(Debug, Clone)]
pub enum LiteralExprValue {
    Number(i64),
    String(String),
    Bool(bool),
}

/// A function-call expression
#[derive(Debug, Clone)]
pub struct FunctionCallExpr {
    pub name: Ident,
    pub args: Vec<Expr>,
    pub args_span: lex::Span,
    pub expr_span: lex::Span,
}

/// An assignment expression
#[derive(Debug, Clone)]
pub struct AssignmentExpr {
    pub place: Box<Expr>,
    pub value: Box<Expr>,
}

/// A binary expression
#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
    pub op_span: lex::Span,
}

/// A unary expression
#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub rhs: Box<Expr>,
    pub op_span: lex::Span,
}

/// A binary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Equal,
    NotEqual,
    LessOrEqual,
    GreaterOrEqual,
    Less,
    Greater,
    Add,
    Sub,
    Mul,
    Div,
}

/// A unary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
}

/// A parser for the source code
pub struct Parser<'a> {
    lexer: lex::Lexer<'a>,
    lookahead: VecDeque<(lex::Span, lex::Token)>,
}

/// An error during parsing with its span
#[derive(Debug, Clone)]
pub struct Error {
    pub span: Option<lex::Span>,
    pub kind: ErrorKind,
}

/// A type of error during parsing
#[derive(Debug, Clone)]
pub enum ErrorKind {
    Lex(lex::ErrorKind),
    UnexpectedToken {
        expected: String,
        got: Option<lex::Token>,
    },
    LetNoValueNoType,
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

/// Pareser implementation
impl Parser<'_> {
    /// Peek the next token without consuming it
    fn peek_token(&mut self) -> Result<Option<&lex::Token>, Error> {
        self.loopahead(0)
    }

    /// Peek n'th token without consuming any tokens
    fn loopahead(&mut self, n: usize) -> Result<Option<&lex::Token>, Error> {
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

    /// Crate an error indicating unexpected token
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

    /// Parse item
    fn next_item(&mut self) -> Result<Option<Item>, Error> {
        match self.peek_token()? {
            Some(lex::Token::Keyword(lex::Keyword::Fn)) => self.next_function().map(Some),
            None => Ok(None),
            _ => self.consume_unexpected_token("an item (function)"),
        }
    }

    /// Parse function
    fn next_function(&mut self) -> Result<Item, Error> {
        self.expect_keyword(lex::Keyword::Fn)?;
        let name = self.next_ident()?;
        self.expect_punct(lex::Punct::LeftParen)?;
        let mut args = Vec::new();
        loop {
            match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::RightParen)) => {
                    self.consume_token()?;
                    break;
                }
                Some(lex::Token::Ident(_)) => {
                    let name = self.next_ident()?;
                    self.expect_punct(lex::Punct::Colon)?;
                    let ty = self.next_ident()?;
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
        self.expect_punct(lex::Punct::Arrow)?;
        let return_ty = self.next_ident()?;
        let body = match self.peek_token()? {
            Some(lex::Token::Punct(lex::Punct::LeftBrace)) => Some(self.next_block_expr()?),
            Some(lex::Token::Punct(lex::Punct::Semicolon)) => {
                self.consume_token()?;
                None
            }
            _ => {
                return self.consume_unexpected_token("function body or ';'");
            }
        };
        Ok(Item::Function(Function {
            name,
            args,
            return_ty,
            body,
        }))
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
                    statements.push(Statement::Empty);
                }
                Some(lex::Token::Keyword(lex::Keyword::Let)) => {
                    statements.push(self.next_let_statement()?);
                }
                _ => match self.next_expr()? {
                    Expr::WithNoBlock(expr_with_no_block) => match self.peek_token()? {
                        Some(lex::Token::Punct(lex::Punct::RightBrace)) => {
                            final_expr = Some(expr_with_no_block);
                            break;
                        }
                        Some(lex::Token::Punct(lex::Punct::Semicolon)) => {
                            self.consume_token()?;
                            statements.push(Statement::ExprWithNoBlock(expr_with_no_block));
                        }
                        _ => {
                            return self.consume_unexpected_token("';' or '}'");
                        }
                    },
                    Expr::WithBlock(expr_with_block) => {
                        statements.push(Statement::ExprWithBlock(expr_with_block));
                    }
                },
            }
        }
        let closing_brace_span = self.expect_punct(lex::Punct::RightBrace)?;
        Ok(BlockExpr {
            statements,
            final_expr,
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
            ty = Some(self.next_ident()?);
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

    /// Parse an experission
    fn next_expr(&mut self) -> Result<Expr, Error> {
        if self.peek_token()? == Some(&lex::Token::Keyword(lex::Keyword::Return)) {
            let (return_keyword_span, _) = self.consume_token()?.unwrap();
            let value = self.next_expr()?;
            Ok(Expr::WithNoBlock(ExprWithNoBlock::Return(ReturnExpr {
                return_keyword_span,
                value: Box::new(value),
            })))
        } else {
            self.next_assigning_expr()
        }
    }

    fn next_assigning_expr(&mut self) -> Result<Expr, Error> {
        let expr = self.next_comp_expr()?;
        if self.peek_token()? == Some(&lex::Token::Punct(lex::Punct::Assign)) {
            self.consume_token()?;
            Ok(Expr::WithNoBlock(ExprWithNoBlock::Assignment(
                AssignmentExpr {
                    place: Box::new(expr),
                    value: Box::new(self.next_comp_expr()?),
                },
            )))
        } else {
            Ok(expr)
        }
    }

    fn next_comp_expr(&mut self) -> Result<Expr, Error> {
        let expr = self.next_additive_expr()?;
        let op = match self.peek_token()? {
            Some(lex::Token::Punct(lex::Punct::CmpEq)) => Some(BinaryOp::Equal),
            Some(lex::Token::Punct(lex::Punct::CmpNeq)) => Some(BinaryOp::NotEqual),
            Some(lex::Token::Punct(lex::Punct::CmpLe)) => Some(BinaryOp::LessOrEqual),
            Some(lex::Token::Punct(lex::Punct::CmpGe)) => Some(BinaryOp::GreaterOrEqual),
            Some(lex::Token::Punct(lex::Punct::CmpL)) => Some(BinaryOp::Less),
            Some(lex::Token::Punct(lex::Punct::CmpG)) => Some(BinaryOp::Greater),
            _ => None,
        };
        match op {
            Some(op) => {
                let (op_span, _) = self.consume_token()?.unwrap();
                Ok(Expr::WithNoBlock(ExprWithNoBlock::Binary(BinaryExpr {
                    op,
                    lhs: Box::new(expr),
                    rhs: Box::new(self.next_additive_expr()?),
                    op_span,
                })))
            }
            None => Ok(expr),
        }
    }

    fn next_additive_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_multiplicative_expr()?;
        loop {
            let op = match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::Plus)) => BinaryOp::Add,
                Some(lex::Token::Punct(lex::Punct::Minus)) => BinaryOp::Sub,
                _ => break,
            };
            let (op_span, _) = self.consume_token()?.unwrap();
            expr = Expr::WithNoBlock(ExprWithNoBlock::Binary(BinaryExpr {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.next_multiplicative_expr()?),
                op_span,
            }));
        }
        Ok(expr)
    }

    fn next_multiplicative_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_base_expr()?;
        loop {
            let op = match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::Star)) => BinaryOp::Mul,
                Some(lex::Token::Punct(lex::Punct::Slash)) => BinaryOp::Div,
                _ => break,
            };
            let (op_span, _) = self.consume_token()?.unwrap();
            expr = Expr::WithNoBlock(ExprWithNoBlock::Binary(BinaryExpr {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.next_base_expr()?),
                op_span,
            }));
        }
        Ok(expr)
    }

    /// Parse a base experission
    fn next_base_expr(&mut self) -> Result<Expr, Error> {
        match self.peek_token()? {
            Some(lex::Token::Ident(_)) => {
                if self.loopahead(1)? == Some(&lex::Token::Punct(lex::Punct::LeftParen)) {
                    self.next_function_call_expr()
                        .map(|expr| Expr::WithNoBlock(ExprWithNoBlock::FunctionCallExpr(expr)))
                } else {
                    self.next_ident()
                        .map(|ident| Expr::WithNoBlock(ExprWithNoBlock::Ident(ident)))
                }
            }
            Some(lex::Token::Literal(_)) => {
                let Some((span, lex::Token::Literal(lit))) = self.consume_token()? else {
                    unreachable!()
                };
                Ok(Expr::WithNoBlock(ExprWithNoBlock::Literal(LiteralExpr {
                    span,
                    value: match lit {
                        lex::Literal::Number(num) => LiteralExprValue::Number(num),
                        lex::Literal::String(str) => LiteralExprValue::String(str),
                    },
                })))
            }
            Some(lex::Token::Keyword(lex::Keyword::True)) => {
                let (span, _) = self.consume_token()?.unwrap();
                Ok(Expr::WithNoBlock(ExprWithNoBlock::Literal(LiteralExpr {
                    span,
                    value: LiteralExprValue::Bool(true),
                })))
            }
            Some(lex::Token::Keyword(lex::Keyword::False)) => {
                let (span, _) = self.consume_token()?.unwrap();
                Ok(Expr::WithNoBlock(ExprWithNoBlock::Literal(LiteralExpr {
                    span,
                    value: LiteralExprValue::Bool(false),
                })))
            }
            Some(lex::Token::Punct(lex::Punct::LeftParen)) => {
                self.consume_token()?;
                let expr = self.next_expr()?;
                self.expect_punct(lex::Punct::RightParen)?;
                Ok(expr)
            }
            Some(lex::Token::Punct(lex::Punct::Minus)) => {
                let (op_span, _) = self.consume_token()?.unwrap();
                let rhs = self.next_base_expr()?;
                Ok(Expr::WithNoBlock(ExprWithNoBlock::Unary(UnaryExpr {
                    op: UnaryOp::Negate,
                    rhs: Box::new(rhs),
                    op_span,
                })))
            }
            Some(lex::Token::Punct(lex::Punct::LeftBrace)) => self
                .next_block_expr()
                .map(|expr| Expr::WithBlock(ExprWithBlock::Block(expr))),
            Some(lex::Token::Keyword(lex::Keyword::If)) => self
                .next_if_expr()
                .map(|expr| Expr::WithBlock(ExprWithBlock::If(expr))),
            Some(lex::Token::Keyword(lex::Keyword::Loop)) => self
                .next_loop_expr()
                .map(|expr| Expr::WithBlock(ExprWithBlock::Loop(expr))),
            _ => self.consume_unexpected_token("expression"),
        }
    }

    /// Parse function call
    fn next_function_call_expr(&mut self) -> Result<FunctionCallExpr, Error> {
        let name = self.next_ident()?;
        let left_paren_span = self.expect_punct(lex::Punct::LeftParen)?;
        let mut args = Vec::new();
        loop {
            match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::RightParen)) => {
                    break;
                }
                _ => {
                    args.push(self.next_expr()?);
                    match self.peek_token()? {
                        Some(lex::Token::Punct(lex::Punct::Comma)) => {
                            self.consume_token()?;
                        }
                        Some(lex::Token::Punct(lex::Punct::RightParen)) => {
                            break;
                        }
                        _ => {
                            return self.consume_unexpected_token("function argument, ',' or ')'");
                        }
                    }
                }
            }
        }
        let right_paren_span = self.expect_punct(lex::Punct::RightParen)?;
        let name_span = name.span;
        Ok(FunctionCallExpr {
            name,
            args,
            args_span: left_paren_span.join(right_paren_span),
            expr_span: name_span.join(right_paren_span),
        })
    }

    /// Parse if expression
    fn next_if_expr(&mut self) -> Result<IfExpr, Error> {
        let if_keyword_span = self.expect_keyword(lex::Keyword::If)?;
        let cond = self.next_expr()?;
        let if_true = self.next_block_expr()?;
        let if_false = if self.peek_token()? == Some(&lex::Token::Keyword(lex::Keyword::Else)) {
            self.consume_token()?;
            Some(self.next_block_expr()?)
        } else {
            None
        };
        Ok(IfExpr {
            cond: Box::new(cond),
            if_true,
            if_false,
            if_keyword_span,
        })
    }

    /// Parse loop expression
    fn next_loop_expr(&mut self) -> Result<LoopExpr, Error> {
        let loop_keyword_span = self.expect_keyword(lex::Keyword::Loop)?;
        let body = self.next_block_expr()?;
        Ok(LoopExpr {
            body,
            loop_keyword_span,
        })
    }
}

impl fmt::Debug for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function(function) => {
                f.write_str("Item::")?;
                function.fmt(f)
            }
        }
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}..{}", self.value, self.span.start, self.span.end)
    }
}
