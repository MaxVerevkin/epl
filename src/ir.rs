mod builder;
pub mod graphviz;
mod types;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::num::NonZeroU64;

use crate::ast;
use crate::common::ArithmeticOp;
use crate::common::CmpOp;
use crate::lex;
use crate::make_entity_id;
pub use types::{Layout, Type, TypeSystem};

/// An intermediate representation of a program
#[derive(Debug)]
pub struct Ir {
    pub typesystem: TypeSystem,
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
        let mut typesystem = TypeSystem::new(8); // TODO: use target arch ptr size!
        let mut function_decls = HashMap::new();

        let mut type_namespace = HashMap::new();
        type_namespace.insert(String::from("void"), Type::Void);
        type_namespace.insert(String::from("bool"), Type::Bool);
        type_namespace.insert(String::from("i32"), Type::I32);
        type_namespace.insert(String::from("u32"), Type::U32);
        type_namespace.insert(String::from("cstr"), Type::CStr);
        type_namespace.insert(String::from("ptr"), Type::OpaquePointer);

        for item in &ast.items {
            match item {
                ast::Item::Function(function) => {
                    function_decls.insert(
                        function.name.value.clone(),
                        FunctionDecl::from_ast(&typesystem, &type_namespace, function)?,
                    );
                }
                ast::Item::Struct(s) => {
                    let name = s.name.value.clone();
                    let s = typesystem.struct_from_ast(&type_namespace, s)?;
                    type_namespace.insert(name, s);
                }
            }
        }

        let mut functions = HashMap::new();
        for item in &ast.items {
            match item {
                ast::Item::Function(function) => {
                    if let Some(body) = &function.body {
                        let decl = &function_decls[&function.name.value];
                        let ir_function =
                            builder::build_function(decl, body, &function_decls, &typesystem, &type_namespace)?;
                        functions.insert(function.name.value.clone(), ir_function);
                    }
                }
                ast::Item::Struct(_) => (),
            }
        }

        Ok(Self {
            typesystem,
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
    fn from_ast(
        typesystem: &TypeSystem,
        type_namespace: &HashMap<String, Type>,
        ast: &ast::Function,
    ) -> Result<Self, Error> {
        Ok(Self {
            name: ast.name.clone(),
            args: ast
                .args
                .iter()
                .map(|a| FunctionArg::from_ast(typesystem, type_namespace, a))
                .collect::<Result<_, _>>()?,
            is_variadic: ast.is_variadic,
            return_ty: ast
                .return_ty
                .as_ref()
                .map(|ty| typesystem.type_from_ast(type_namespace, ty))
                .transpose()?
                .unwrap_or(Type::Void),
        })
    }
}

impl FunctionArg {
    /// Create a function argument representation from its AST
    fn from_ast(
        typesystem: &TypeSystem,
        type_namespace: &HashMap<String, Type>,
        ast: &ast::FunctionArg,
    ) -> Result<Self, Error> {
        Ok(Self {
            name: ast.name.clone(),
            ty: typesystem.type_from_ast(type_namespace, &ast.ty)?,
        })
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
        visit(&mut order, &mut HashSet::new(), &self.basic_blokcs, self.entry);
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
    pub layout: Layout,
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
    Cmp { op: CmpOp, lhs: Value, rhs: Value },
    Arithmetic { op: ArithmeticOp, lhs: Value, rhs: Value },
    Not { value: Value },
    OffsetPtr { ptr: Value, offset: i64 },
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
#[derive(Clone)]
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
#[derive(Debug, Clone)]
pub enum Constant {
    Undefined(Type),
    Void,
    Bool(bool),
    String(String),
    Number { data: i64, bits: u8, signed: bool },
}

impl Constant {
    /// Get the type of this constant
    pub fn ty(&self) -> Type {
        match self {
            Self::Undefined(ty) => *ty,
            Self::Void => Type::Void,
            Self::Bool(_) => Type::Bool,
            Self::String(_) => Type::CStr,
            Self::Number { data: _, bits, signed } => match *bits {
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
