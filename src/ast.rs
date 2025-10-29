use std::collections::VecDeque;

use crate::lex;

/// The abstract sytax tree representation of a source code file
#[derive(Debug, Clone)]
pub struct Ast {
    pub items: Vec<Item>,
}

/// A top level item
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Ident {
    pub span: lex::Span,
    pub value: String,
}

/// A block expression
#[derive(Debug, Clone)]
pub struct BlockExpr {
    pub statements: Vec<Statement>,
    pub final_expr: Option<ExprWithNoBlock>,
}

/// An `if` expression
#[derive(Debug, Clone)]
pub struct IfExpr {
    pub cond: Box<Expr>,
    pub if_true: BlockExpr,
    pub if_false: Option<BlockExpr>,
}

/// A `loop` expression
#[derive(Debug, Clone)]
pub struct LoopExpr {
    pub body: BlockExpr,
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
pub struct LetStatement {
    name: Ident,
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
    Literal(LiteralExpr),
    FunctionCallExpr(FunctionCallExpr),
    Ident(Ident),
    Binary(BinaryExpr),
}

/// An expression which can be a statement on its own
#[derive(Debug, Clone)]
pub enum ExprWithBlock {
    Block(BlockExpr),
    If(IfExpr),
    Loop(LoopExpr),
}

/// A literal expression
#[derive(Debug, Clone)]
pub enum LiteralExpr {
    Number(i64),
    String(String),
}

/// A function-call expression
#[derive(Debug, Clone)]
pub struct FunctionCallExpr {
    pub name: Ident,
    pub args: Vec<Expr>,
}

/// A binary expression
#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
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
                    expected: format!("identifier"),
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
        self.expect_punct(lex::Punct::LeftBrace)?;
        let mut statements = Vec::new();
        let mut final_expr = None;
        loop {
            match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::RightBrace)) => {
                    self.consume_token()?;
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
                            self.consume_token()?;
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
        Ok(BlockExpr {
            statements,
            final_expr,
        })
    }

    /// Parse let statement
    fn next_let_statement(&mut self) -> Result<Statement, Error> {
        self.expect_keyword(lex::Keyword::Let)?;
        let name = self.next_ident()?;
        self.expect_punct(lex::Punct::Semicolon)?;
        Ok(Statement::Let(LetStatement { name }))
    }

    /// Parse an experission
    fn next_expr(&mut self) -> Result<Expr, Error> {
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
                self.consume_token()?;
                Ok(Expr::WithNoBlock(ExprWithNoBlock::Binary(BinaryExpr {
                    op,
                    lhs: Box::new(expr),
                    rhs: Box::new(self.next_additive_expr()?),
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
            self.consume_token()?;
            expr = Expr::WithNoBlock(ExprWithNoBlock::Binary(BinaryExpr {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.next_multiplicative_expr()?),
            }));
        }
        Ok(expr)
    }

    fn next_multiplicative_expr(&mut self) -> Result<Expr, Error> {
        let mut expr = self.next_base_expr_or_with_block()?;
        loop {
            let op = match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::Star)) => BinaryOp::Mul,
                Some(lex::Token::Punct(lex::Punct::Slash)) => BinaryOp::Div,
                _ => break,
            };
            self.consume_token()?;
            expr = Expr::WithNoBlock(ExprWithNoBlock::Binary(BinaryExpr {
                op,
                lhs: Box::new(expr),
                rhs: Box::new(self.next_base_expr_or_with_block()?),
            }));
        }
        Ok(expr)
    }

    /// Parse a base experission
    fn next_base_expr_or_with_block(&mut self) -> Result<Expr, Error> {
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
                let Some((_, lex::Token::Literal(lit))) = self.consume_token()? else {
                    unreachable!()
                };
                Ok(Expr::WithNoBlock(ExprWithNoBlock::Literal(match lit {
                    lex::Literal::Number(num) => LiteralExpr::Number(num),
                    lex::Literal::String(str) => LiteralExpr::String(str),
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
        self.expect_punct(lex::Punct::LeftParen)?;
        let mut args = Vec::new();
        loop {
            match self.peek_token()? {
                Some(lex::Token::Punct(lex::Punct::RightParen)) => {
                    self.consume_token()?;
                    break;
                }
                _ => {
                    args.push(self.next_expr()?);
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
            }
        }
        Ok(FunctionCallExpr { name, args })
    }

    /// Parse if expression
    fn next_if_expr(&mut self) -> Result<IfExpr, Error> {
        self.expect_keyword(lex::Keyword::If)?;
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
        })
    }

    /// Parse loop expression
    fn next_loop_expr(&mut self) -> Result<LoopExpr, Error> {
        self.expect_keyword(lex::Keyword::Loop)?;
        let body = self.next_block_expr()?;
        Ok(LoopExpr { body })
    }
}
