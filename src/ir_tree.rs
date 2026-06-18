mod checkers;
mod dump;
mod evaluator;
mod lower_ast;
mod mono;
mod opt;
mod types;
mod visit;

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

pub use types::*;

use crate::common::{ArithmeticOp, BinaryOp, CmpOp, PtrSize};
use crate::context::Context;
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

    /// Create a new 'type mismatch' error with a given span
    pub fn expr_type_mismatch(expected: Type, found: Type, span: lex::Span) -> Self {
        Self {
            span: Some(span),
            message: format!("expected expr of type {}, found {}", expected.render(), found.render()),
        }
    }

    pub fn unknown_annotation(annotation: &ast::Annotation) -> Self {
        Self {
            span: Some(annotation.span()),
            message: format!("unknown annotation: {:?}", annotation.ident.value),
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
make_entity_id!(StructFieldId, "struct_field_{}");

pub struct Module<'ctx> {
    pub ctx: Context<'ctx>,
    pub functions: BTreeMap<FunctionId, Function<'ctx>>,
}

impl<'ctx> Module<'ctx> {
    /// Construct an IR_TREE from AST
    pub fn from_ast(ctx: Context<'ctx>, ast: &ast::Ast) -> Result<Self, Error> {
        let mut module = Self {
            ctx,
            functions: BTreeMap::new(),
        };

        let mut functions_namespace = HashMap::new();

        let mut types_scope = TypesScope::default();
        for (name, ty) in [
            ("unit", ctx.types().unit),
            ("bool", ctx.types().bool),
            ("i8", ctx.types().i8),
            ("u8", ctx.types().u8),
            ("i32", ctx.types().i32),
            ("u32", ctx.types().u32),
            ("i64", ctx.types().i64),
            ("u64", ctx.types().u64),
            ("isize", ctx.types().isize),
            ("usize", ctx.types().usize),
            ("ptr", ctx.types().opaque_ptr),
        ] {
            types_scope
                .by_name
                .insert(name.to_owned(), TypeConstructor::NonGeneric(ty));
        }

        // Pass 1 - collect ADT skeletons
        let mut structs = HashMap::new();
        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(_) => (),
                ast::ItemKind::Struct(s_def) => {
                    if types_scope.lookup(&s_def.name.value).is_some() {
                        return Err(Error::new("type with this name already exists").with_span(s_def.name.span));
                    }
                    if let Some(annotation) = item.annotations.first() {
                        return Err(Error::unknown_annotation(annotation));
                    }
                    let fields = s_def
                        .fields
                        .iter()
                        .map(|f| (f.name.clone(), StructFieldId::new()))
                        .collect();
                    let struct_ = ctx.new_struct_unchecked(s_def.name.clone(), s_def.type_parameters.len(), fields);
                    structs.insert(s_def.name.value.as_str(), struct_);
                    let type_constructor = if s_def.type_parameters.is_empty() {
                        TypeConstructor::NonGeneric(Type::new(
                            ctx,
                            TypeInfo::Struct {
                                struct_,
                                type_arguments: TypeArguments::new(ctx, &[]),
                            },
                        ))
                    } else {
                        TypeConstructor::Struct(struct_)
                    };
                    types_scope.by_name.insert(s_def.name.value.clone(), type_constructor);
                }
                ast::ItemKind::Enum(_e_def) => {
                    unimplemented!("enum support is not here yet");
                }
            }
        }

        // Pass 2 - Lower ADS's fields types
        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(_) => (),
                ast::ItemKind::Struct(s_def) => {
                    types_scope.push();
                    let struct_ = structs[&*s_def.name.value];
                    types_scope.insert_ast_type_parameters(
                        ctx,
                        TypeParameterOwner::Struct(struct_),
                        &s_def.type_parameters,
                    )?;
                    for ((_, field_id), field_ast) in struct_.info().fields.iter().zip(&s_def.fields) {
                        let field_ty = type_from_ast(ctx, &types_scope, &field_ast.ty)?;
                        ctx.register_struct_field_type(*field_id, field_ty, struct_);
                    }
                    types_scope.pop();
                }
                ast::ItemKind::Enum(_e_def) => {
                    unimplemented!("enum support is not here yet");
                }
            }
        }

        // Pass 3 - reject recursive structs w/o indirection
        for &struct_ in structs.values() {
            struct_.reject_recursive_without_indirection(ctx)?;
        }

        // // Pass 3 - Resolve layouts
        // for &id in &struct_ids {
        //     module.typesystem.resolve_layout(Type::Struct(id), &mut Vec::new())?;
        // }

        // Pass 4 - collect function declarations
        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(function) => {
                    let decl = lower_ast::lower_function_decl(ctx, function, &item.annotations, &mut types_scope)?;
                    if functions_namespace
                        .insert(function.name.value.clone(), decl.id)
                        .is_some()
                    {
                        return Err(Error::new("function with this name already exists").with_span(function.name.span));
                    }
                    module.functions.insert(decl.id, decl);
                }
                ast::ItemKind::Struct(_) | ast::ItemKind::Enum(_) => (),
            }
        }

        // Pass 5 - lower function bodies
        for item in &ast.items {
            match &item.kind {
                ast::ItemKind::Function(function) => {
                    let function_id = functions_namespace[&function.name.value];
                    if let Some(body) = &function.body {
                        let decl = &module.functions[&function_id];
                        let body = lower_ast::lower_function_body(
                            ctx,
                            decl,
                            function,
                            body,
                            &functions_namespace,
                            &module.functions,
                            &mut types_scope,
                        )?;
                        module.functions.get_mut(&function_id).unwrap().body = Some(body);
                    }
                }
                ast::ItemKind::Struct(_) | ast::ItemKind::Enum(_) => (),
            }
        }

        for function_id in module.functions.keys() {
            checkers::run_checkers(*function_id, &module)?;
        }

        for function in module.functions.values_mut() {
            if let Some(body) = &mut function.body {
                opt::BasicOptVisitor(ctx).visit_expr(body);
            }
        }

        let to_mono = mono::collect_for_monomorphization(&module);
        mono::monomorphize(&mut module, &to_mono);

        for function_id in module.functions.keys().copied().collect::<Vec<_>>() {
            // TODO: this is ridiculously inefficient O(n^2), for something that could potentially be O(n).

            fn get_first_comptime_expr<'a, 'ctx>(function: &'a Function<'ctx>) -> Option<&'a Expr<'ctx>> {
                struct Visitor<'a, 'ctx>(Option<&'a Expr<'ctx>>);
                impl<'a, 'ctx> ExprVisitor<'a, 'ctx> for Visitor<'a, 'ctx> {
                    fn visit_expr(&mut self, expr: &'a Expr<'ctx>) {
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

            fn set_first_comptime_expr<'ctx>(ctx: Context<'ctx>, function: &mut Function<'ctx>, value: Constant<'ctx>) {
                struct Visitor<'ctx>(Context<'ctx>, Option<Constant<'ctx>>);
                impl<'a, 'ctx> ExprVisitorMut<'a, 'ctx> for Visitor<'ctx> {
                    fn visit_expr(&mut self, expr: &'a mut Expr<'ctx>) {
                        if matches!(expr.kind, ExprKind::Comptime(_)) {
                            if let Some(value) = self.1.take() {
                                *expr = Expr::new_const(self.0, value);
                            }
                        } else {
                            expr.visit_children_mut(self);
                        }
                    }
                }
                let mut v = Visitor(ctx, Some(value));
                v.visit_expr(function.body.as_mut().unwrap());
            }

            while let Some(expr) = get_first_comptime_expr(&module.functions[&function_id]) {
                let evaluated = evaluator::eval_comptime_expr(expr, &module)?;
                set_first_comptime_expr(ctx, module.functions.get_mut(&function_id).unwrap(), evaluated);
            }
        }

        Ok(module)
    }

    /// Dump the contents of this module in a human-readable representation.
    ///
    /// Note: the representation is not stable and should only be used for
    /// demonstration/debugging purposes
    pub fn dump(&self) -> String {
        dump::dump(self)
    }
}

#[derive(Debug)]
pub struct Function<'ctx> {
    pub id: FunctionId,
    pub name_ident: ast::Ident,
    pub mangled_name: String,
    pub debug_name: String,
    pub monomorphized_for: Option<(FunctionId, TypeArguments<'ctx>)>,
    pub type_parameters: usize,
    pub args: Vec<(String, Type<'ctx>)>,
    pub return_ty: Type<'ctx>,
    pub is_variadic: bool,
    pub is_pure: bool,
    pub body: Option<Expr<'ctx>>,
}

#[derive(Debug, Clone)]
pub struct Expr<'ctx> {
    pub ty: Type<'ctx>,
    pub span: Option<lex::Span>,
    pub kind: ExprKind<'ctx>,
}

#[derive(Debug, Clone)]
pub struct Place<'ctx> {
    pub ty: Type<'ctx>,
    pub span: Option<lex::Span>,
    pub kind: PlaceKind<'ctx>,
}

impl<'ctx> Place<'ctx> {
    pub fn dummy(ctx: Context<'ctx>) -> Self {
        Self {
            ty: ctx.types().unit,
            span: None,
            kind: PlaceKind::Variable(VariableId::DUMMY),
        }
    }

    fn var(var: VariableId, ty: Type<'ctx>) -> Self {
        Self {
            ty,
            span: None,
            kind: PlaceKind::Variable(var),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExprKind<'ctx> {
    Const(Constant<'ctx>),
    ConstString(String),

    Load(Place<'ctx>),
    Field(Box<Expr<'ctx>>, StructFieldId),
    ArrayElement(Box<Expr<'ctx>>, Box<Expr<'ctx>>),

    Store(Place<'ctx>, Box<Expr<'ctx>>),
    GetPointer(Place<'ctx>),

    Argument(usize),
    Block(BlockExpr<'ctx>),
    Return(Box<Expr<'ctx>>),
    Break(LoopId, Box<Expr<'ctx>>),
    Continue(LoopId),
    Arithmetic(ArithmeticOp, Box<Expr<'ctx>>, Box<Expr<'ctx>>),
    InPlaceArithmetic(ArithmeticOp, Place<'ctx>, Box<Expr<'ctx>>),
    Cmp(CmpOp, Box<Expr<'ctx>>, Box<Expr<'ctx>>),
    If { cond: Box<Expr<'ctx>>, if_true: Box<Expr<'ctx>>, if_false: Box<Expr<'ctx>> },
    Loop(LoopId, Box<Expr<'ctx>>),
    ArrayInitializer(Vec<Expr<'ctx>>),
    StructInitializer(Vec<(StructFieldId, Expr<'ctx>)>),
    FunctionCall(FunctionId, TypeArguments<'ctx>, Vec<Expr<'ctx>>),
    Cast(Box<Expr<'ctx>>),
    Not(Box<Expr<'ctx>>),
    Comptime(Box<Expr<'ctx>>),
}

#[derive(Debug, Clone)]
pub enum PlaceKind<'ctx> {
    Dereference(Box<Expr<'ctx>>),
    Variable(VariableId),
    Field(Box<Place<'ctx>>, StructFieldId),
    ArrayElement(Box<Place<'ctx>>, Box<Expr<'ctx>>),
}

#[derive(Debug, Clone)]
pub enum Constant<'ctx> {
    Undefined(Type<'ctx>),
    Null(Type<'ctx>),
    Unit,
    Bool(bool),
    I8(i8),
    U8(u8),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    ISize(i128),
    USize(i128),
    Array(Type<'ctx>, Vec<Self>),
    Struct(Type<'ctx>, Vec<Self>),
}

impl<'ctx> Constant<'ctx> {
    pub fn ty(&self, ctx: Context<'ctx>) -> Type<'ctx> {
        match self {
            Self::Undefined(ty) | Self::Null(ty) | Self::Array(ty, _) | Self::Struct(ty, _) => *ty,
            Self::Unit => ctx.types().unit,
            Self::Bool(_) => ctx.types().bool,
            Self::I8(_) => ctx.types().i8,
            Self::U8(_) => ctx.types().u8,
            Self::I32(_) => ctx.types().i32,
            Self::U32(_) => ctx.types().u32,
            Self::I64(_) => ctx.types().i64,
            Self::U64(_) => ctx.types().u64,
            Self::ISize(_) => ctx.types().isize,
            Self::USize(_) => ctx.types().usize,
        }
    }

    /// Returns `None` if the number cannot fit into provided `IntType`
    pub fn int(ctx: Context, number: i128, ty: IntType) -> Option<Self> {
        Some(match ty {
            IntType::I8 => Self::I8(number.try_into().ok()?),
            IntType::U8 => Self::U8(number.try_into().ok()?),
            IntType::I32 => Self::I32(number.try_into().ok()?),
            IntType::U32 => Self::U32(number.try_into().ok()?),
            IntType::I64 => Self::I64(number.try_into().ok()?),
            IntType::U64 => Self::U64(number.try_into().ok()?),
            IntType::ISize => {
                match ctx.ptr_size() {
                    PtrSize::_64 => {
                        if i64::try_from(number).is_err() {
                            return None;
                        }
                    }
                }
                Self::ISize(number)
            }
            IntType::USize => {
                match ctx.ptr_size() {
                    PtrSize::_64 => {
                        if u64::try_from(number).is_err() {
                            return None;
                        }
                    }
                }
                Self::ISize(number)
            }
        })
    }

    pub fn with_erased_isize_usize(&self, ctx: Context<'ctx>) -> Cow<'_, Self> {
        match self {
            Self::ISize(num) => match ctx.ptr_size() {
                PtrSize::_64 => Cow::Owned(Self::I64(*num as i64)),
            },
            Self::USize(num) => match ctx.ptr_size() {
                PtrSize::_64 => Cow::Owned(Self::U64(*num as u64)),
            },
            _ => Cow::Borrowed(self),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockExpr<'ctx> {
    pub variables: Vec<VariableDeclaration<'ctx>>,
    pub exprs: Vec<Expr<'ctx>>,
}

#[derive(Debug, Clone)]
pub struct VariableDeclaration<'ctx> {
    pub id: VariableId,
    pub ty: Type<'ctx>,
    pub debug_name: String,
}

impl<'ctx> Expr<'ctx> {
    pub fn unit(ctx: Context<'ctx>) -> Self {
        Self {
            ty: ctx.types().unit,
            span: None,
            kind: ExprKind::Const(Constant::Unit),
        }
    }

    pub fn into_place(self) -> Option<Place<'ctx>> {
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

    pub fn expect_place(self) -> Result<Place<'ctx>, Error> {
        let span = self.span.unwrap();
        self.into_place()
            .ok_or_else(|| Error::new("expected a place expression").with_span(span))
    }

    fn get_var(var: VariableId, ty: Type<'ctx>) -> Self {
        Self {
            ty,
            span: None,
            kind: ExprKind::Load(Place::var(var, ty)),
        }
    }

    fn set_var(ctx: Context<'ctx>, var: VariableId, expr: Expr<'ctx>) -> Self {
        Self {
            ty: ctx.types().unit,
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

    fn new_const(ctx: Context<'ctx>, constant: Constant<'ctx>) -> Self {
        Self {
            ty: constant.ty(ctx),
            span: None,
            kind: ExprKind::Const(constant),
        }
    }

    fn const_bool(ctx: Context<'ctx>, bool: bool) -> Self {
        Self {
            ty: ctx.types().bool,
            span: None,
            kind: ExprKind::Const(Constant::Bool(bool)),
        }
    }
}
