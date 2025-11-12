mod builder;
pub mod graphviz;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
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
    pub is_variadic: bool,
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
            is_variadic: ast.is_variadic,
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
    OpaquePointer,
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
            "ptr" => Ok(Self::OpaquePointer),
            other => Err(Error::new(format!("unknown type {other:?}")).with_span(ast.span)),
        }
    }

    /// The byte size of this type
    fn size(self) -> u64 {
        match self {
            Type::Never | Type::Void => 0,
            Type::Bool => 1,
            Type::I32 | Type::U32 => 4,
            Type::CStr | Type::OpaquePointer => 8, // TODO: use target platforms pointer size
        }
    }

    /// The byte alignment of this type
    fn align(self) -> u64 {
        match self {
            Type::Never | Type::Void | Type::Bool => 1,
            Type::I32 | Type::U32 => 4,
            Type::CStr | Type::OpaquePointer => 8, // TODO: use target platforms pointer size
        }
    }

    /// Returns `true` if this data type is an integer
    fn is_int(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Returns `true` if this data type is a signed integer
    fn is_signed_int(self) -> bool {
        matches!(self, Self::I32)
    }

    /// Returns `true` if this data type is an unsigned integer
    fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U32)
    }

    /// Combines two types into one, handling the Never type
    ///
    /// 1. If `self` or `other` is Never, the other type is returned.
    /// 2. If both are Never, Never is returned.
    /// 3. Ohterwise require types to be equal, and return the type.
    fn comine_ignoring_never(self, other: Self) -> Option<Self> {
        if self == Self::Never {
            Some(other)
        } else if other == Self::Never {
            Some(self)
        } else {
            (self == other).then_some(self)
        }
    }
}

/// An intermediate representation of a function
#[derive(Debug)]
pub struct Function {
    pub allocas: Vec<Alloca>,
    pub entry: BasicBlockId,
    pub basic_blokcs: HashMap<BasicBlockId, BasicBlock>,
}

impl Function {
    /// Return the basick blocks IDs in post order
    pub fn postorder(&self) -> Vec<BasicBlockId> {
        fn visit(
            order: &mut Vec<BasicBlockId>,
            visited: &mut HashSet<BasicBlockId>,
            basic_blocks: &HashMap<BasicBlockId, BasicBlock>,
            cur: BasicBlockId,
        ) {
            if !visited.insert(cur) {
                return;
            }

            for succ in basic_blocks[&cur].terminator.successors() {
                visit(order, visited, basic_blocks, succ);
            }

            order.push(cur);
        }

        let mut order = Vec::new();
        visit(
            &mut order,
            &mut HashSet::new(),
            &self.basic_blokcs,
            self.entry,
        );
        order
    }
}

make_entity_id!(BasicBlockId, "bb_{}");

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DefinitionId(NonZeroU64, Type);

impl DefinitionId {
    pub fn new(ty: Type) -> Self {
        use std::sync::atomic;
        static NEXT: atomic::AtomicU64 = atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, atomic::Ordering::SeqCst);
        Self(NonZeroU64::new(id).unwrap(), ty)
    }

    pub fn ty(self) -> Type {
        self.1
    }
}

impl Hash for DefinitionId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Debug for DefinitionId {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "def_{}: {:?}", self.0.get(), self.1)
    }
}

/// A static allocation slot
#[derive(Debug, Clone, Copy)]
pub struct Alloca {
    pub definition_id: DefinitionId,
    pub size: u64,
    pub align: u64,
}

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
    pub kind: InstructionKind,
}

/// The kind of instruction / operation
#[derive(Debug)]
pub enum InstructionKind {
    Load { ptr: Value },
    Store { ptr: Value, value: Value },
    FunctionCall { name: String, args: Vec<Value> },
    CmpSL { lhs: Value, rhs: Value },
    CmpUL { lhs: Value, rhs: Value },
    Add { lhs: Value, rhs: Value },
    Sub { lhs: Value, rhs: Value },
    Mul { lhs: Value, rhs: Value },
}

/// The terminator of a basic block
#[derive(Debug)]
pub enum Terminator {
    Jump {
        to: BasicBlockId,
        args: Vec<Value>,
    },
    CondJump {
        cond: Value,
        if_true: BasicBlockId,
        if_true_args: Vec<Value>,
        if_false: BasicBlockId,
        if_false_args: Vec<Value>,
    },
    Return {
        value: Value,
    },
    Unreachable,
}

// impl InstructionKind {
//     pub fn visit_values(&self, mut cb: impl FnMut(&Value)) {
//         match self {
//             Self::Load { ptr } => cb(ptr),
//             Self::Store { ptr, value } => {
//                 cb(ptr);
//                 cb(value);
//             }
//             Self::FunctionCall { name: _, args } => {
//                 for arg in args {
//                     cb(arg);
//                 }
//             }
//             Self::CmpSL { lhs, rhs }
//             | Self::CmpUL { lhs, rhs }
//             | Self::Add { lhs, rhs }
//             | Self::Sub { lhs, rhs }
//             | Self::Mul { lhs, rhs } => {
//                 cb(lhs);
//                 cb(rhs);
//             }
//         }
//     }

//     pub fn visit_used_definitions(&self, mut cb: impl FnMut(DefinitionId)) {
//         self.visit_values(|value| match value {
//             Value::Definition(definition_id) => cb(*definition_id),
//             Value::Constant(_) => (),
//         });
//     }
// }

impl Terminator {
    /// Return the list of this block's successors
    pub fn successors(&self) -> Vec<BasicBlockId> {
        match self {
            Self::Jump { to, args: _ } => vec![*to],
            Self::CondJump {
                cond: _,
                if_true,
                if_true_args: _,
                if_false,
                if_false_args: _,
            } => vec![*if_true, *if_false],
            Self::Return { value: _ } | Self::Unreachable => vec![],
        }
    }
}

/// An abstract value
pub enum Value {
    Definition(DefinitionId),
    Constant(Constant),
}

impl Value {
    /// Get the type of this constant
    pub fn ty(&self) -> Type {
        match self {
            Self::Definition(definition_id) => definition_id.ty(),
            Self::Constant(constant) => constant.ty(),
        }
    }
}

/// A primitive constant
#[derive(Debug)]
pub enum Constant {
    Void,
    Bool(bool),
    String(String),
    Number { data: i64, bits: u8, signed: bool },
}

impl Constant {
    /// Get the type of this constant
    pub fn ty(&self) -> Type {
        match self {
            Self::Void => Type::Void,
            Self::Bool(_) => Type::Bool,
            Self::String(_) => Type::CStr,
            Self::Number {
                data: _,
                bits,
                signed,
            } => match *bits {
                32 => match *signed {
                    true => Type::I32,
                    false => Type::U32,
                },
                _ => unreachable!(),
            },
        }
    }
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
        write!(f, "{:?} <- {:?}", self.definition_id, self.kind)
    }
}
