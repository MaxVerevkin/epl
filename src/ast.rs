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
    ExprWithNoBlock(ExprWithNoBlock),
    ExprWithBlock(ExprWithBlock),
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

/// A parser for the source code
pub struct Parser<'a> {
    lexer: lex::Lexer<'a>,
}

/// An error during parsing
#[derive(Debug, Clone)]
pub enum Error {
    Lex(lex::Error),
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given source code
    pub fn new(src: &'a str) -> Self {
        Self {
            lexer: lex::Lexer::new(src),
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
    fn next_item(&mut self) -> Result<Option<Item>, Error> {
        todo!()
    }
}
