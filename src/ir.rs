mod builder;
pub mod graphviz;

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU64;

use crate::ast;
use crate::lex;
use crate::make_entity_id;

/// An intermediate representation of a program
#[derive(Debug)]
pub struct Ir {
    pub function_decls: HashMap<String, FunctionDecl>,
    pub functions: HashMap<String, Function>,
}

/// An error during IR construction and typechecking
#[derive(Debug)]
pub struct Error {
    pub span: Option<lex::Span>,
    pub message: String,
}

impl Error {
    /// Create a new error with the given message
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            span: None,
            message: msg.into(),
        }
    }

    /// Create a new 'type missmatch' error with a given span
    pub fn expr_type_missmatch(expected: Type, found: Type, span: lex::Span) -> Self {
        Self {
            span: Some(span),
            message: format!("expectd expr of type {expected:?}, found {found:?}"),
        }
    }

    /// Assign a span to this error
    pub fn with_span(mut self, span: lex::Span) -> Self {
        self.span = Some(span);
        self
    }
}

impl Ir {
    /// Construct an IR from AST
    pub fn from_ast(ast: &ast::Ast) -> Result<Self, Error> {
        let mut function_decls = HashMap::new();

        for item in &ast.items {
            match item {
                ast::Item::Function(function) => {
                    function_decls.insert(
                        function.name.value.clone(),
                        FunctionDecl::from_ast(function)?,
                    );
                }
            }
        }

        let mut functions = HashMap::new();
        for item in &ast.items {
            match item {
                ast::Item::Function(function) => {
                    if let Some(body) = &function.body {
                        let decl = &function_decls[&function.name.value];
                        let ir_function = builder::build_function(decl, body, &function_decls)?;
                        functions.insert(function.name.value.clone(), ir_function);
                    }
                }
            }
        }

        Ok(Self {
            function_decls,
            functions,
        })
    }
}

/// A function declaration
#[derive(Debug)]
pub struct FunctionDecl {
    pub name: ast::Ident,
    pub args: Vec<FunctionArg>,
    pub return_ty: Type,
}

/// A function declaration argument
#[derive(Debug)]
pub struct FunctionArg {
    pub name: ast::Ident,
    pub ty: Type,
}

impl FunctionDecl {
    /// Construct a function declaration from its AST
    fn from_ast(ast: &ast::Function) -> Result<Self, Error> {
        Ok(Self {
            name: ast.name.clone(),
            args: ast
                .args
                .iter()
                .map(FunctionArg::from_ast)
                .collect::<Result<_, _>>()?,
            return_ty: Type::from_ast(&ast.return_ty)?,
        })
    }
}

impl FunctionArg {
    /// Create a function argument representation from its AST
    fn from_ast(ast: &ast::FunctionArg) -> Result<Self, Error> {
        Ok(Self {
            name: ast.name.clone(),
            ty: Type::from_ast(&ast.ty)?,
        })
    }
}

/// The set of data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Never,
    Void,
    Bool,
    I32,
    U32,
    CStr,
}

impl Type {
    /// Construct a data type from its AST
    fn from_ast(ast: &ast::Ident) -> Result<Self, Error> {
        match &*ast.value {
            "!" => Ok(Self::Never),
            "void" => Ok(Self::Void),
            "bool" => Ok(Self::Bool),
            "i32" => Ok(Self::I32),
            "u32" => Ok(Self::U32),
            "cstr" => Ok(Self::CStr),
            other => Err(Error::new(format!("unknown type {other:?}")).with_span(ast.span)),
        }
    }

    /// Returns `true` if this data type is an integer
    fn is_int(self) -> bool {
        matches!(self, Self::I32 | Self::U32)
    }
}

/// An intermediate representation of a function
#[derive(Debug)]
pub struct Function {
    pub allocas: HashMap<DefinitionId, Type>,
    pub entry: BasicBlockId,
    pub basic_blokcs: HashMap<BasicBlockId, BasicBlock>,
}

make_entity_id!(BasicBlockId, "bb_{}");

make_entity_id!(DefinitionId, "def_{}");

/// A basic block
#[derive(Debug)]
pub struct BasicBlock {
    pub args: Vec<DefinitionId>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

/// A basic instruction
pub struct Instruction {
    pub definition_id: DefinitionId,
    pub ty: Type,
    pub kind: InstructionKind,
}

/// The kind of instruction / operation
#[derive(Debug)]
pub enum InstructionKind {
    Load { ptr: Value },
    Store { ptr: Value, value: Value },
    FunctionCall { name: String, args: Vec<Value> },
    CmpL { lhs: Value, rhs: Value },
    Add { lhs: Value, rhs: Value },
    Sub { lhs: Value, rhs: Value },
}

/// The terminator of a basic block
#[derive(Debug)]
pub enum Terminator {
    Jump {
        to: BasicBlockId,
    },
    CondJump {
        cond: Value,
        if_true: BasicBlockId,
        if_false: BasicBlockId,
    },
    Return {
        value: Value,
    },
    Unreachable,
}

/// An abstract value
pub enum Value {
    Definition(DefinitionId),
    Constant(Constant),
}

/// A primitive constant
#[derive(Debug)]
pub enum Constant {
    Void,
    Bool(bool),
    String(String),
    Number(i64),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Definition(def) => def.fmt(f),
            Self::Constant(c) => c.fmt(f),
        }
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}: {:?} <- {:?}",
            self.definition_id, self.ty, self.kind
        )
    }
}
