pub mod graphviz;
mod lower_ir_tree;
mod opt;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;

use crate::common::{ArithmeticOp, CmpOp, Layout};
use crate::{ir_tree, lex, make_entity_id};

/// An intermediate representation of a program
#[derive(Debug)]
pub struct Ir {
    pub functions: Vec<Function>,
}

/// The set of data types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Unit,
    Bool,
    I8,
    I32,
    I64,
    Ptr,
    Struct(Vec<Self>),
    Array(Box<Self>, u64),
}

impl Type {
    fn layout(&self, module: &ir_tree::Module) -> Layout {
        match self {
            Type::Unit => Layout { size: 0, align: 1 },
            Type::Bool | Type::I8 => Layout { size: 1, align: 1 },
            Type::I32 => Layout { size: 4, align: 4 },
            Type::I64 => Layout { size: 8, align: 8 },
            Type::Ptr => Layout {
                size: module.typesystem.ptr_size(),
                align: module.typesystem.ptr_size(),
            },
            Type::Struct(fields) => {
                let mut layout = Layout { size: 0, align: 1 };
                for field in fields {
                    let field_layout = field.layout(module);
                    layout.align = layout.align.max(field_layout.align);
                    layout.size = layout.size.next_multiple_of(field_layout.align);
                    layout.size += field_layout.size;
                }
                layout.size = layout.size.next_multiple_of(layout.align);
                layout
            }
            Type::Array(element, length) => {
                let element_layout = element.layout(module);
                Layout {
                    size: element_layout.size * length,
                    align: element_layout.align,
                }
            }
        }
    }

    fn is_zst(&self) -> bool {
        match self {
            Self::Unit => true,
            Self::Bool | Self::I8 | Self::I32 | Self::I64 | Self::Ptr => false,
            Self::Struct(items) => items.iter().all(Self::is_zst),
            Self::Array(element, length) => *length == 0 || element.is_zst(),
        }
    }

    fn array_element_type(&self) -> Option<&Self> {
        match self {
            Self::Array(element, _) => Some(element),
            _ => None,
        }
    }

    pub fn int_bits(&self) -> Option<u32> {
        match self {
            Type::I8 => Some(8),
            Type::I32 => Some(32),
            Type::I64 => Some(64),
            Type::Unit | Type::Bool | Type::Ptr | Type::Struct(_) | Type::Array(_, _) => None,
        }
    }
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

    /// Assign a span to this error
    pub fn with_span(mut self, span: lex::Span) -> Self {
        self.span = Some(span);
        self
    }
}

impl Ir {
    /// Construct an IR from IR_TREE
    pub fn from_ir_tree(ir_tree: &ir_tree::Module) -> Result<Self, Error> {
        let mut this = Self {
            functions: ir_tree
                .functions
                .values()
                .map(|function| lower_ir_tree::lower_function(function, ir_tree))
                .collect::<Result<_, _>>()?,
        };

        for function in &mut this.functions {
            opt::basic_passes(function);
        }

        Ok(this)
    }
}

/// A function declaration
#[derive(Debug)]
pub struct Function {
    pub mangled_name: String,
    pub args: Vec<Type>,
    pub is_variadic: bool,
    #[expect(unused)]
    pub never_returs: bool,
    pub return_ty: Type,
    pub body: Option<FunctionBody>,
}

/// An intermediate representation of a function
#[derive(Debug)]
pub struct FunctionBody {
    pub allocas: Vec<Alloca>,
    pub entry: BasicBlockId,
    pub basic_blokcs: HashMap<BasicBlockId, BasicBlock>,
}

impl FunctionBody {
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

#[derive(Clone, PartialEq, Eq)]
pub struct DefinitionId(NonZeroU64, Type);

impl DefinitionId {
    pub fn new(ty: Type) -> Self {
        use std::sync::atomic;
        static NEXT: atomic::AtomicU64 = atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, atomic::Ordering::SeqCst);
        Self(NonZeroU64::new(id).unwrap(), ty)
    }

    pub fn ty(&self) -> &Type {
        &self.1
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
#[derive(Debug, Clone)]
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
    Cmp { op: CmpOp, signed: bool, lhs: Value, rhs: Value },
    Arithmetic { op: ArithmeticOp, signed: bool, lhs: Value, rhs: Value },
    Not { value: Value },
    OffsetPtr { ptr: Value, offset: Value },
    Zext { int: Value },
    Sext { int: Value },
    Truncate { int: Value },
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
    Return(Value),
    Unreachable,
}

impl InstructionKind {
    pub fn visit_operands_mut(&mut self, mut cb: impl FnMut(&mut Value)) {
        match self {
            Self::Load { ptr: x }
            | Self::Not { value: x }
            | Self::Zext { int: x }
            | Self::Sext { int: x }
            | Self::Truncate { int: x } => cb(x),
            Self::Store { ptr: x, value: y }
            | Self::Arithmetic {
                op: _,
                signed: _,
                lhs: x,
                rhs: y,
            }
            | Self::Cmp {
                op: _,
                signed: _,
                lhs: x,
                rhs: y,
            }
            | Self::OffsetPtr { ptr: x, offset: y } => {
                cb(x);
                cb(y);
            }
            Self::FunctionCall { name: _, args } => {
                for arg in args {
                    cb(arg);
                }
            }
        }
    }
}

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
            Self::Return(_) | Self::Unreachable => vec![],
        }
    }

    pub fn visit_operands_mut(&mut self, mut cb: impl FnMut(&mut Value)) {
        match self {
            Self::Jump { to: _, args } => {
                for arg in args {
                    cb(arg);
                }
            }
            Self::CondJump {
                cond,
                if_true: _,
                if_true_args,
                if_false: _,
                if_false_args,
            } => {
                cb(cond);
                for arg in if_true_args {
                    cb(arg);
                }
                for arg in if_false_args {
                    cb(arg);
                }
            }
            Self::Return(_) | Self::Unreachable => (),
        }
    }
}

/// An abstract value
#[derive(Clone, PartialEq, Eq)]
pub enum Value {
    Zst,
    Undefined(Type),
    Bool(bool),
    String(String),
    Number { data: i64, ty: Type },
    Definition(DefinitionId),
}

impl Value {
    /// Get the type of this constant
    pub fn ty(&self) -> Type {
        match self {
            Self::Zst => Type::Unit,
            Self::Undefined(ty) => ty.clone(),
            Self::Bool(_) => Type::Bool,
            Self::String(_) => Type::Ptr,
            Self::Number { data: _, ty } => ty.clone(),
            Self::Definition(definition_id) => definition_id.ty().clone(),
        }
    }

    /// Create a new constant with a given value
    pub fn new_i64(data: i64) -> Self {
        Self::Number { data, ty: Type::I64 }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zst => f.write_str("ZST"),
            Self::Undefined(_) => f.write_str("undefined"),
            Self::Bool(bool) => bool.fmt(f),
            Self::String(str) => str.fmt(f),
            Self::Number { data, ty: _ } => data.fmt(f),
            Self::Definition(def) => def.fmt(f),
        }
    }
}

impl fmt::Debug for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} <- {:?}", self.definition_id, self.kind)
    }
}
