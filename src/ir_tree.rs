mod dump;
mod lower_ast;
pub mod types;

use std::collections::HashMap;

use crate::{ast, common::BinaryOp, lex, make_entity_id};
use types::{IntType, Type, TypeSystem};

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
    pub functions: HashMap<FunctionId, Function>,
    pub typesystem: TypeSystem,
}

impl Module {
    /// Construct an IR from AST
    pub fn from_ast(ast: &ast::Ast) -> Result<Self, Error> {
        let mut typesystem = TypeSystem::new(8); // TODO: use target arch ptr size!
        let mut functions_namespace = HashMap::new();
        let mut functions = HashMap::new();

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
            match item {
                ast::Item::Function(function) => {
                    let decl = Function::decl_from_ast(&mut typesystem, &type_namespace, function)?;
                    functions_namespace.insert(function.name.value.clone(), decl.id);
                    functions.insert(decl.id, decl);
                }
                ast::Item::Struct(s) => {
                    let name = s.name.value.clone();
                    let s = typesystem.struct_from_ast(&type_namespace, s)?;
                    type_namespace.insert(name, s);
                }
            }
        }

        for item in &ast.items {
            match item {
                ast::Item::Function(function) => {
                    if let Some(body) = &function.body {
                        let function_id = functions_namespace[&function.name.value];
                        let decl = &functions[&function_id];
                        let body = lower_ast::build_function_body(
                            decl,
                            body,
                            &functions_namespace,
                            &functions,
                            &mut typesystem,
                            &type_namespace,
                        )?;
                        functions.get_mut(&function_id).unwrap().body = Some(body);
                    }
                }
                ast::Item::Struct(_) => (),
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
    pub body: Option<BlockExpr>,
}

impl Function {
    /// Construct a function declaration from its AST
    fn decl_from_ast(
        typesystem: &mut TypeSystem,
        type_namespace: &HashMap<String, Type>,
        ast: &ast::Function,
    ) -> Result<Self, Error> {
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
            body: None,
        })
    }
}

#[derive(Debug)]
pub enum Expr {
    R(RExpr),
    L(LExpr),
}

#[derive(Debug)]
pub struct RExpr {
    pub ty: Type,
    pub span: Option<lex::Span>,
    pub kind: RExprKind,
}

#[derive(Debug)]
pub struct LExpr {
    pub ty: Type,
    pub span: Option<lex::Span>,
    pub kind: LExprKind,
}

#[derive(Debug)]
pub enum RExprKind {
    Undefined,
    ConstUnit,
    ConstNumber(i64),
    ConstString(String),
    ConstBool(bool),

    Field(Box<RExpr>, String),
    ArrayElement(Box<RExpr>, Box<Expr>),

    Store(Box<LExpr>, Box<Expr>),
    GetPointer(Box<LExpr>),

    Block(BlockExpr),
    Return(Box<Expr>),
    Break(LoopId, Box<Expr>),
    BinOp(BinaryOp, Box<Expr>, Box<Expr>),
    If { cond: Box<Expr>, if_true: Box<Expr>, if_false: Option<Box<Expr>> },
    Loop(LoopId, Box<Expr>),
    ArrayInitializer(Vec<Expr>),
    StructInitializer(Vec<(String, Expr)>),
    FunctionCall(FunctionId, Vec<Expr>),
    Cast(Box<Expr>),
    Not(Box<Expr>),
}

#[derive(Debug)]
pub enum LExprKind {
    Dereference(Box<Expr>),
    Variable(VariableId),
    Field(Box<LExpr>, String),
    ArrayElement(Box<LExpr>, Box<Expr>),
}

impl Expr {
    pub fn expect_lvalue(self) -> Result<LExpr, Error> {
        match self {
            Self::L(lvalue) => Ok(lvalue),
            Self::R(rvalue) => Err(Error::new("expected an l-value expression").with_span(rvalue.span.unwrap())),
        }
    }
}

#[derive(Debug)]
pub struct BlockExpr {
    pub variables: Vec<(VariableId, Type)>,
    pub exprs: Vec<Expr>,
}

impl Expr {
    const UNIT: Self = Self::R(RExpr {
        ty: Type::Unit,
        span: None,
        kind: RExprKind::ConstUnit,
    });

    fn ty(&self) -> Type {
        match self {
            Self::R(e) => e.ty,
            Self::L(e) => e.ty,
        }
    }

    fn span(&self) -> Option<lex::Span> {
        match self {
            Self::R(e) => e.span,
            Self::L(e) => e.span,
        }
    }

    fn get_var(var: VariableId, ty: Type) -> Self {
        Self::L(LExpr {
            ty,
            span: None,
            kind: LExprKind::Variable(var),
        })
    }

    fn set_var(var: VariableId, expr: Expr) -> Self {
        Self::R(RExpr {
            ty: Type::Unit,
            span: None,
            kind: RExprKind::Store(
                Box::new(LExpr {
                    ty: expr.ty(),
                    span: None,
                    kind: LExprKind::Variable(var),
                }),
                Box::new(expr),
            ),
        })
    }

    fn const_number(number: i64, ty: Type) -> Self {
        Self::R(RExpr {
            ty,
            span: None,
            kind: RExprKind::ConstNumber(number),
        })
    }

    fn const_bool(bool: bool) -> Self {
        Self::R(RExpr {
            ty: Type::Bool,
            span: None,
            kind: RExprKind::ConstBool(bool),
        })
    }
}

impl LExpr {
    fn dereference(expr: Expr, ty: Type) -> Self {
        Self {
            ty,
            span: None,
            kind: LExprKind::Dereference(Box::new(expr)),
        }
    }
}
