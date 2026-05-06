mod checkers;
mod dump;
mod evaluator;
mod lower_ast;
mod opt;
mod types;
mod visit;

use std::collections::{BTreeMap, BTreeSet, HashMap};

pub use types::{IntType, Type, TypeId, TypeSystem};

use crate::common::{ArithmeticOp, BinaryOp, CmpOp};
use crate::ir_tree::visit::{ExprVisitor, ExprVisitorMut};
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
    /// Construct an IR_TREE from AST
    pub fn from_ast(ast: &ast::Ast) -> Result<Self, Error> {
        let mut module = Self {
            functions: BTreeMap::new(),
            typesystem: TypeSystem::new(8), // TODO: use target arch ptr size!
        };

        let mut functions_namespace = HashMap::new();

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
                ast::ItemKind::Function(_) => (),
                ast::ItemKind::Struct(s_def) => {
                    let name = s_def.name.value.clone();
                    let s = module
                        .typesystem
                        .struct_from_ast(&type_namespace, s_def, &item.annotations)?;
                    if type_namespace.insert(name, s).is_some() {
                        return Err(Error::new("type with this name already exists").with_span(s_def.name.span));
                    }
                }
            }
        }

        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(function) => {
                    let decl =
                        Function::decl_from_ast(&mut module.typesystem, &type_namespace, function, &item.annotations)?;
                    if functions_namespace
                        .insert(function.name.value.clone(), decl.id)
                        .is_some()
                    {
                        return Err(Error::new("function with this name already exists").with_span(decl.name.span));
                    }
                    module.functions.insert(decl.id, decl);
                }
                ast::ItemKind::Struct(_) => (),
            }
        }

        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(function) => {
                    let function_id = functions_namespace[&function.name.value];
                    if let Some(body) = &function.body {
                        let decl = &module.functions[&function_id];
                        let body = lower_ast::lower_function_body(
                            decl,
                            body,
                            &functions_namespace,
                            &module.functions,
                            &mut module.typesystem,
                            &type_namespace,
                        )?;
                        module.functions.get_mut(&function_id).unwrap().body = Some(body);
                    }
                }
                ast::ItemKind::Struct(_) => (),
            }
        }

        for function_id in module.functions.keys() {
            checkers::run_checkers(*function_id, &module)?;
        }

        for function in module.functions.values_mut() {
            if let Some(body) = &mut function.body {
                opt::BasicOptVisitor.visit_expr(body);
            }
        }

        for function_id in module.functions.keys().copied().collect::<Vec<_>>() {
            // TODO: this is ridiculusly inefficient O(n^2), for something that could potentially be O(n).

            fn get_first_comptime_expr(function: &Function) -> Option<&Expr> {
                struct Visitor<'a>(Option<&'a Expr>);
                impl<'a> ExprVisitor<'a> for Visitor<'a> {
                    fn visit_expr(&mut self, expr: &'a Expr) {
                        match &expr.kind {
                            ExprKind::Comptime(expr) => {
                                if self.0.is_none() {
                                    self.0 = Some(expr);
                                }
                            }
                            _ => expr.visit_children(self),
                        }
                    }
                }
                let mut v = Visitor(None);
                v.visit_expr(function.body.as_ref()?);
                v.0
            }

            fn set_first_comptime_expr(function: &mut Function, value: Constant) {
                struct Visitor(Option<Constant>);
                impl ExprVisitorMut for Visitor {
                    fn visit_expr(&mut self, expr: &mut Expr) {
                        if matches!(expr.kind, ExprKind::Comptime(_)) {
                            if let Some(value) = self.0.take() {
                                *expr = Expr::new_const(value);
                            }
                        } else {
                            expr.visit_children_mut(self);
                        }
                    }
                }
                let mut v = Visitor(Some(value));
                v.visit_expr(function.body.as_mut().unwrap());
            }

            while let Some(expr) = get_first_comptime_expr(&module.functions[&function_id]) {
                let evaluated = evaluator::eval_comptime_expr(expr, &module)?;
                set_first_comptime_expr(module.functions.get_mut(&function_id).unwrap(), evaluated);
            }
        }

        Ok(module)
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

        let mut args: Vec<(String, Type)> = Vec::new();
        for arg in &ast.args {
            if args.iter().any(|x| x.0 == arg.name.value) {
                return Err(Error::new("argument with this name already exists").with_span(arg.name.span));
            }
            args.push((
                arg.name.value.clone(),
                typesystem.type_from_ast(type_namespace, &arg.ty)?,
            ));
        }

        Ok(Self {
            id: FunctionId::new(),
            name: ast.name.clone(),
            args,
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
    Const(Constant),
    ConstString(String),

    Load(Place),
    Field(Box<Expr>, String),
    ArrayElement(Box<Expr>, Box<Expr>),

    Store(Place, Box<Expr>),
    GetPointer(Place),

    Argument(usize),
    Block(BlockExpr),
    Return(Box<Expr>),
    Break(LoopId, Box<Expr>),
    Continue(LoopId),
    Arithmetic(ArithmeticOp, Box<Expr>, Box<Expr>),
    InPlaceArithmetic(ArithmeticOp, Place, Box<Expr>),
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
    If { cond: Box<Expr>, if_true: Box<Expr>, if_false: Box<Expr> },
    Loop(LoopId, Box<Expr>),
    ArrayInitializer(Vec<Expr>),
    StructInitializer(Vec<(String, Expr)>),
    FunctionCall(FunctionId, Vec<Expr>),
    Cast(Box<Expr>),
    Not(Box<Expr>),
    Comptime(Box<Expr>),
}

#[derive(Debug)]
pub enum PlaceKind {
    Dereference(Box<Expr>),
    Variable(VariableId),
    Field(Box<Place>, String),
    ArrayElement(Box<Place>, Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum Constant {
    Undefined(Type),
    Unit,
    Bool(bool),
    I8(i8),
    U8(u8),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    Array(TypeId, Vec<Self>),
}

impl Constant {
    pub fn ty(&self) -> Type {
        match self {
            Self::Undefined(ty) => *ty,
            Self::Unit => Type::Unit,
            Self::Bool(_) => Type::Bool,
            Self::I8(_) => Type::Int(IntType::I8),
            Self::U8(_) => Type::Int(IntType::U8),
            Self::I32(_) => Type::Int(IntType::I32),
            Self::U32(_) => Type::Int(IntType::U32),
            Self::I64(_) => Type::Int(IntType::I64),
            Self::U64(_) => Type::Int(IntType::U64),
            Self::Array(element_ty, elements) => Type::Array {
                element: *element_ty,
                length: elements.len() as u64,
            },
        }
    }

    /// Returns `None` if the number cannot fit into provided `IntType`
    pub fn int(number: u128, ty: IntType) -> Option<Self> {
        Some(match ty {
            IntType::I8 => Self::I8(number.try_into().ok()?),
            IntType::U8 => Self::U8(number.try_into().ok()?),
            IntType::I32 => Self::I32(number.try_into().ok()?),
            IntType::U32 => Self::U32(number.try_into().ok()?),
            IntType::I64 => Self::I64(number.try_into().ok()?),
            IntType::U64 => Self::U64(number.try_into().ok()?),
        })
    }

    /// Returns `None` if the number cannot fit into provided `IntType`
    pub fn int_signed(number: i128, ty: IntType) -> Option<Self> {
        Some(match ty {
            IntType::I8 => Self::I8(number.try_into().ok()?),
            IntType::U8 => Self::U8(number.try_into().ok()?),
            IntType::I32 => Self::I32(number.try_into().ok()?),
            IntType::U32 => Self::U32(number.try_into().ok()?),
            IntType::I64 => Self::I64(number.try_into().ok()?),
            IntType::U64 => Self::U64(number.try_into().ok()?),
        })
    }
}

#[derive(Debug)]
pub struct BlockExpr {
    pub variables: Vec<VariableDeclaration>,
    pub exprs: Vec<Expr>,
}

#[derive(Debug)]
pub struct VariableDeclaration {
    pub id: VariableId,
    pub ty: Type,
    pub debug_name: String,
}

impl Expr {
    const UNIT: Self = Self {
        ty: Type::Unit,
        span: None,
        kind: ExprKind::Const(Constant::Unit),
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
            ExprKind::Const(..)
            | ExprKind::ConstString(..)
            | ExprKind::Store(..)
            | ExprKind::GetPointer(..)
            | ExprKind::Argument(..)
            | ExprKind::Block(..)
            | ExprKind::Return(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Arithmetic(..)
            | ExprKind::InPlaceArithmetic(..)
            | ExprKind::Cmp(..)
            | ExprKind::If { .. }
            | ExprKind::Loop(..)
            | ExprKind::ArrayInitializer(..)
            | ExprKind::StructInitializer(..)
            | ExprKind::FunctionCall(..)
            | ExprKind::Cast(..)
            | ExprKind::Not(..)
            | ExprKind::Comptime(..) => None,
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
            kind: ExprKind::Load(Place::var(var, ty)),
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

    fn new_const(constant: Constant) -> Self {
        Self {
            ty: constant.ty(),
            span: None,
            kind: ExprKind::Const(constant),
        }
    }

    fn const_bool(bool: bool) -> Self {
        Self {
            ty: Type::Bool,
            span: None,
            kind: ExprKind::Const(Constant::Bool(bool)),
        }
    }
}

impl Place {
    fn var(var: VariableId, ty: Type) -> Self {
        Self {
            ty,
            span: None,
            kind: PlaceKind::Variable(var),
        }
    }
}
