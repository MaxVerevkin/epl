mod checkers;
mod dump;
mod lower_ast;
mod opt;
mod types;
mod visit;

use std::collections::{BTreeMap, BTreeSet, HashMap};

pub use types::{IntType, Type, TypeSystem};

use crate::common::{ArithmeticOp, BinaryOp, CmpOp};
use crate::ir_tree::visit::ExprVisitorMut;
use crate::{ast, lex, make_entity_id};

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

make_entity_id!(FunctionId, "fn_{}");
make_entity_id!(VariableId, "var_{}");
make_entity_id!(LoopId, "loop_{}");

#[derive(Debug)]
pub struct Module {
    pub functions: BTreeMap<FunctionId, Function>,
    pub typesystem: TypeSystem,
}

impl Module {
    /// Construct an IR from AST
    pub fn from_ast(ast: &ast::Ast) -> Result<Self, Error> {
        let mut typesystem = TypeSystem::new(8); // TODO: use target arch ptr size!
        let mut functions_namespace = HashMap::new();
        let mut functions = BTreeMap::new();

        let mut type_namespace = HashMap::new();
        type_namespace.insert(String::from("unit"), Type::Unit);
        type_namespace.insert(String::from("bool"), Type::Bool);
        type_namespace.insert(String::from("i8"), Type::Int(IntType::I8));
        type_namespace.insert(String::from("u8"), Type::Int(IntType::U8));
        type_namespace.insert(String::from("i32"), Type::Int(IntType::I32));
        type_namespace.insert(String::from("u32"), Type::Int(IntType::U32));
        type_namespace.insert(String::from("i64"), Type::Int(IntType::I64));
        type_namespace.insert(String::from("u64"), Type::Int(IntType::U64));
        type_namespace.insert(String::from("ptr"), Type::OPAQUE_PTR);

        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(function) => {
                    let decl = Function::decl_from_ast(&mut typesystem, &type_namespace, function, &item.annotations)?;
                    functions_namespace.insert(function.name.value.clone(), decl.id);
                    functions.insert(decl.id, decl);
                }
                ast::ItemKind::Struct(s) => {
                    let name = s.name.value.clone();
                    let s = typesystem.struct_from_ast(&type_namespace, s, &item.annotations)?;
                    type_namespace.insert(name, s);
                }
            }
        }

        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(function) => {
                    if let Some(body) = &function.body {
                        let function_id = functions_namespace[&function.name.value];
                        let decl = &functions[&function_id];
                        let mut body = lower_ast::lower_function_body(
                            decl,
                            body,
                            &functions_namespace,
                            &functions,
                            &mut typesystem,
                            &type_namespace,
                        )?;
                        if decl.is_pure {
                            checkers::purity_check(&body, &functions)?;
                        }
                        opt::BasicOptVisitor.visit_expr(&mut body);
                        functions.get_mut(&function_id).unwrap().body = Some(body);
                    }
                }
                ast::ItemKind::Struct(_) => (),
            }
        }

        Ok(Self { typesystem, functions })
    }

    /// Dump the contents of this module in a human-readable representation.
    ///
    /// Note: the representation is not stable and should only be used for
    /// demonstation/debugging purposes
    pub fn dump(&self) -> String {
        dump::dump(self)
    }
}

#[derive(Debug)]
pub struct Function {
    pub id: FunctionId,
    pub name: ast::Ident,
    pub args: Vec<(String, Type)>,
    pub return_ty: Type,
    pub is_variadic: bool,
    pub is_pure: bool,
    pub body: Option<Expr>,
}

impl Function {
    /// Construct a function declaration from its AST
    fn decl_from_ast(
        typesystem: &mut TypeSystem,
        type_namespace: &HashMap<String, Type>,
        ast: &ast::Function,
        annotations: &BTreeSet<ast::Annotation>,
    ) -> Result<Self, Error> {
        let mut is_pure = false;
        for annotation in annotations {
            match annotation.ident.value.as_str() {
                "pure" => is_pure = true,
                other => {
                    return Err(Error::new(format!("unknown annotation: {other:?}")).with_span(annotation.span()));
                }
            }
        }
        Ok(Self {
            id: FunctionId::new(),
            name: ast.name.clone(),
            args: ast
                .args
                .iter()
                .map(|a| Ok((a.name.value.clone(), typesystem.type_from_ast(type_namespace, &a.ty)?)))
                .collect::<Result<_, _>>()?,
            return_ty: ast
                .return_ty
                .as_ref()
                .map(|ty| typesystem.type_from_ast(type_namespace, ty))
                .transpose()?
                .unwrap_or(Type::Unit),
            is_variadic: ast.is_variadic,
            is_pure,
            body: None,
        })
    }
}

#[derive(Debug)]
pub struct Expr {
    pub ty: Type,
    pub span: Option<lex::Span>,
    pub kind: ExprKind,
}

#[derive(Debug)]
pub struct Place {
    pub ty: Type,
    pub span: Option<lex::Span>,
    pub kind: PlaceKind,
}

impl Place {
    pub const DUMMY: Self = Self {
        ty: Type::Never,
        span: None,
        kind: PlaceKind::Variable(VariableId::DUMMY),
    };
}

#[derive(Debug)]
pub enum ExprKind {
    Undefined,
    ConstUnit,
    ConstNumber(i64),
    ConstString(String),
    ConstBool(bool),

    Load(Place),
    Field(Box<Expr>, String),
    ArrayElement(Box<Expr>, Box<Expr>),

    Store(Place, Box<Expr>),
    GetPointer(Place),

    Argument(String),
    Block(BlockExpr),
    Return(Box<Expr>),
    Break(LoopId, Box<Expr>),
    Arithmetic(ArithmeticOp, Box<Expr>, Box<Expr>),
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
    If { cond: Box<Expr>, if_true: Box<Expr>, if_false: Box<Expr> },
    Loop(LoopId, Box<Expr>),
    ArrayInitializer(Vec<Expr>),
    StructInitializer(Vec<(String, Expr)>),
    FunctionCall(FunctionId, Vec<Expr>),
    Cast(Box<Expr>),
    Not(Box<Expr>),
}

#[derive(Debug)]
pub enum PlaceKind {
    Dereference(Box<Expr>),
    Variable(VariableId),
    Field(Box<Place>, String),
    ArrayElement(Box<Place>, Box<Expr>),
}

#[derive(Debug)]
pub struct BlockExpr {
    pub variables: Vec<(VariableId, Type)>,
    pub exprs: Vec<Expr>,
}

impl Expr {
    const UNIT: Self = Self {
        ty: Type::Unit,
        span: None,
        kind: ExprKind::ConstUnit,
    };

    pub fn into_place(self) -> Option<Place> {
        match self.kind {
            ExprKind::Load(place) => Some(place),
            ExprKind::Field(place, field) => Some(Place {
                ty: self.ty,
                span: self.span,
                kind: PlaceKind::Field(Box::new(place.into_place()?), field),
            }),
            ExprKind::ArrayElement(place, index) => Some(Place {
                ty: self.ty,
                span: self.span,
                kind: PlaceKind::ArrayElement(Box::new(place.into_place()?), index),
            }),
            ExprKind::Undefined
            | ExprKind::ConstUnit
            | ExprKind::ConstNumber(..)
            | ExprKind::ConstString(..)
            | ExprKind::ConstBool(..)
            | ExprKind::Store(..)
            | ExprKind::GetPointer(..)
            | ExprKind::Argument(..)
            | ExprKind::Block(..)
            | ExprKind::Return(..)
            | ExprKind::Break(..)
            | ExprKind::Arithmetic(..)
            | ExprKind::Cmp(..)
            | ExprKind::If { .. }
            | ExprKind::Loop(..)
            | ExprKind::ArrayInitializer(..)
            | ExprKind::StructInitializer(..)
            | ExprKind::FunctionCall(..)
            | ExprKind::Cast(..)
            | ExprKind::Not(..) => None,
        }
    }

    pub fn expect_place(self) -> Result<Place, Error> {
        let span = self.span.unwrap();
        self.into_place()
            .ok_or_else(|| Error::new("expected a place expression").with_span(span))
    }

    fn get_var(var: VariableId, ty: Type) -> Self {
        Self {
            ty,
            span: None,
            kind: ExprKind::Load(Place {
                ty,
                span: None,
                kind: PlaceKind::Variable(var),
            }),
        }
    }

    fn set_var(var: VariableId, expr: Expr) -> Self {
        Self {
            ty: Type::Unit,
            span: None,
            kind: ExprKind::Store(
                Place {
                    ty: expr.ty,
                    span: None,
                    kind: PlaceKind::Variable(var),
                },
                Box::new(expr),
            ),
        }
    }

    fn const_number(number: i64, ty: Type) -> Self {
        Self {
            ty,
            span: None,
            kind: ExprKind::ConstNumber(number),
        }
    }

    fn const_bool(bool: bool) -> Self {
        Self {
            ty: Type::Bool,
            span: None,
            kind: ExprKind::ConstBool(bool),
        }
    }
}

impl Place {
    fn dereference(expr: Expr, ty: Type) -> Self {
        Self {
            ty,
            span: None,
            kind: PlaceKind::Dereference(Box::new(expr)),
        }
    }
}
