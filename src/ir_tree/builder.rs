use std::collections::HashMap;

use super::*;
use crate::{
    ast,
    common::{ArithmeticOp, CmpOp},
};

/// Construct an IR of a function from its AST
pub fn build_function_body(
    decl: &Function,
    body: &ast::BlockExpr,
    functions_namespace: &HashMap<String, FunctionId>,
    functions: &HashMap<FunctionId, Function>,
    typesystem: &mut TypeSystem,
    type_namespace: &HashMap<String, Type>,
) -> Result<BlockExpr, Error> {
    if decl.is_variadic {
        return Err(Error::new("defining variadic functions is not supported").with_span(decl.name.span));
    }

    let mut builder = FunctionLoweringCtx::new(decl, functions_namespace, functions, typesystem, type_namespace);

    let body_eval = builder.lower_block_expr(body, Some(decl.return_ty))?;

    Ok(BlockExpr {
        variables: builder.args,
        statements: Vec::new(),
        final_expr: Some(Box::new(body_eval)),
    })
}

/// A function's AST -> IR_TREE lowering context
struct FunctionLoweringCtx<'a> {
    decl: &'a Function,
    functions_namespace: &'a HashMap<String, FunctionId>,
    functions: &'a HashMap<FunctionId, Function>,
    args: Vec<(VariableId, Type)>,
    typesystem: &'a mut TypeSystem,
    type_namespace: &'a HashMap<String, Type>,
    scope: Scope,
}

/// A lexical scope
#[derive(Default)]
struct Scope {
    variables: HashMap<String, (VariableId, Type)>,
    loop_context: Option<LoopContext>,
    parent: Option<Box<Self>>,
}

#[derive(Clone, Copy)]
struct LoopContext {
    break_from: LoopId,
    break_used_with_type: Option<Type>,
    expect_type: Option<Type>,
}

impl Scope {
    /// Create a nested scope
    fn push(&mut self) {
        let parent = std::mem::take(self);
        self.parent = Some(Box::new(parent));
    }

    /// Pop the latest nested scope
    fn pop(&mut self) {
        *self = *self.parent.take().unwrap();
    }

    /// Lookup a variable by its name, recursively traversind the list of scopes
    fn lookup_variable(&self, name: &str) -> Option<(VariableId, Type)> {
        if let Some(definition_id) = self.variables.get(name) {
            return Some(*definition_id);
        }
        if let Some(parent) = &self.parent {
            return parent.lookup_variable(name);
        }
        None
    }

    /// Recursively lookup a loop context
    fn loop_context(&mut self) -> Option<&mut LoopContext> {
        if let Some(ctx) = &mut self.loop_context {
            return Some(ctx);
        }
        if let Some(parent) = &mut self.parent {
            return parent.loop_context();
        }
        None
    }
}

struct LowerLoopBodyResult {
    loop_id: LoopId,
    body: Expr,
    break_used_with_type: Option<Type>,
}

impl<'a> FunctionLoweringCtx<'a> {
    /// Create a new builder context
    fn new(
        decl: &'a Function,
        functions_namespace: &'a HashMap<String, FunctionId>,
        functions: &'a HashMap<FunctionId, Function>,
        typesystem: &'a mut TypeSystem,
        type_namespace: &'a HashMap<String, Type>,
    ) -> Self {
        let mut this = Self {
            decl,
            functions_namespace,
            functions,
            args: Vec::new(),
            typesystem,
            type_namespace,
            scope: Scope::default(),
        };

        for (name, ty) in &decl.args {
            let var_id = VariableId::new();
            this.scope.variables.insert(name.clone(), (var_id, *ty));
            this.args.push((var_id, *ty));
        }

        this
    }

    /// lower an expression
    fn lower_expr(&mut self, expr: &ast::Expr, expect_type: Option<Type>) -> Result<Expr, Error> {
        fn coalesce_types(a: Type, b: Type) -> Type {
            if a == Type::Never { b } else { a }
        }

        let span = Some(expr.span());

        match expr {
            ast::Expr::Block(block_expr) => self.lower_block_expr(block_expr, expect_type),
            ast::Expr::If(if_expr) => {
                let expect_type = match (expect_type, if_expr.if_false.is_some()) {
                    (None | Some(Type::Void), false) => Some(Type::Void),

                    (_, false) => {
                        return Err(Error::new(format!(
                            "if experession expected to evalueate to type {expect_type:?}, so it must have an else branch"
                        )).with_span(if_expr.if_keyword_span));
                    }

                    (expect_type, true) => expect_type,
                };

                let cond = self.lower_expr(&if_expr.cond, Some(Type::Bool))?;

                let if_true = self.lower_block_expr(&if_expr.if_true, expect_type)?;

                let expect_type = if if_true.ty() != Type::Never {
                    Some(if_true.ty())
                } else {
                    expect_type
                };

                let if_false = if_expr
                    .if_false
                    .as_ref()
                    .map(|expr| self.lower_block_expr(expr, expect_type))
                    .transpose()?;

                Ok(Expr::R(RExpr {
                    ty: coalesce_types(if_true.ty(), if_false.as_ref().map_or(Type::Void, |expr| expr.ty())),
                    span,
                    kind: RExprKind::If {
                        cond: Box::new(cond),
                        if_true: Box::new(if_true),
                        if_false: if_false.map(Box::new),
                    },
                }))
            }
            ast::Expr::Loop(loop_expr) => {
                let lowered_body = self.lower_loop_body(&loop_expr.body, expect_type)?;
                Ok(Expr::R(RExpr {
                    ty: lowered_body.break_used_with_type.unwrap_or(Type::Never),
                    span,
                    kind: RExprKind::Loop(lowered_body.loop_id, Box::new(lowered_body.body)),
                }))
            }
            ast::Expr::While(while_expr) => {
                // transform
                //
                // while <cond> { $body }
                //
                // to
                //
                // loop {
                //     if <cond> { $body } else { break }
                // }
                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::expr_type_missmatch(expect_type, Type::Void, expr.span()));
                }
                let cond = self.lower_expr(&while_expr.cond, Some(Type::Bool))?;
                let lowered_body = self.lower_loop_body(&while_expr.body, Some(Type::Void))?;
                Ok(Expr::R(RExpr {
                    ty: Type::Void,
                    span,
                    kind: RExprKind::Loop(
                        lowered_body.loop_id,
                        Box::new(Expr::R(RExpr {
                            ty: Type::Void,
                            span,
                            kind: RExprKind::If {
                                cond: Box::new(cond),
                                if_true: Box::new(lowered_body.body),
                                if_false: Some(Box::new(Expr::R(RExpr {
                                    ty: Type::Never,
                                    span: None,
                                    kind: RExprKind::Break(lowered_body.loop_id, None),
                                }))),
                            },
                        })),
                    ),
                }))
            }
            ast::Expr::For(e) => {
                // transform
                //
                // for <var> in <expr_from>..<expr_to> { $body }
                //
                // to
                //
                // {
                //     let $var = <expr_from>
                //     let $target = <expr_to>
                //     loop {
                //         if $var < $target {
                //             let <var> = $var
                //             { $body }
                //             $var += 1
                //         } else {
                //             break
                //         }
                //     }
                // }
                //
                // TODO: shadowed target will no longer be needed when mutability is implemented

                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::expr_type_missmatch(expect_type, Type::Void, expr.span()));
                }

                let ast::Expr::Range(range_expr) = &*e.iterator else {
                    return Err(
                        Error::new("only range exprs (e.g. 'a..b') are supported as iterator in 'for' yet")
                            .with_span(e.iterator.span()),
                    );
                };

                // Only i32 is supported for now
                let var_type = Type::Int(IntType::I32);
                let from_expr = self.lower_expr(&range_expr.from, Some(var_type))?;
                let to_expr = self.lower_expr(&range_expr.to, Some(var_type))?;

                let var_id = VariableId::new();
                let target_id = VariableId::new();
                let shadowed_var_id = VariableId::new();

                self.scope.push();
                self.scope
                    .variables
                    .insert(e.i.value.clone(), (shadowed_var_id, var_type));
                let lowered_body = self.lower_loop_body(&e.body, Some(Type::Void))?;
                self.scope.pop();

                Ok(Expr::R(RExpr {
                    ty: Type::Void,
                    span,
                    kind: RExprKind::Block(BlockExpr {
                        variables: vec![(var_id, var_type), (target_id, var_type), (shadowed_var_id, var_type)],
                        statements: vec![
                            Expr::set_var(var_id, from_expr),
                            Expr::set_var(target_id, to_expr),
                            Expr::R(RExpr {
                                ty: Type::Void,
                                span: None,
                                kind: RExprKind::Loop(
                                    lowered_body.loop_id,
                                    Box::new(Expr::R(RExpr {
                                        ty: Type::Void,
                                        span: None,
                                        kind: RExprKind::If {
                                            cond: Box::new(Expr::R(RExpr {
                                                ty: Type::Bool,
                                                span: None,
                                                kind: RExprKind::BinOp(
                                                    BinaryOp::Cmp(CmpOp::Less),
                                                    Box::new(Expr::get_var(var_id, var_type)),
                                                    Box::new(Expr::get_var(target_id, var_type)),
                                                ),
                                            })),
                                            if_true: Box::new(Expr::R(RExpr {
                                                ty: Type::Void,
                                                span: None,
                                                kind: RExprKind::Block(BlockExpr {
                                                    variables: Vec::new(),
                                                    statements: vec![
                                                        Expr::set_var(shadowed_var_id, Expr::get_var(var_id, var_type)),
                                                        lowered_body.body,
                                                        Expr::set_var(
                                                            var_id,
                                                            Expr::R(RExpr {
                                                                ty: var_type,
                                                                span: None,
                                                                kind: RExprKind::BinOp(
                                                                    BinaryOp::Arithmetic(ArithmeticOp::Add),
                                                                    Box::new(Expr::get_var(var_id, var_type)),
                                                                    Box::new(Expr::const_number(1, var_type)),
                                                                ),
                                                            }),
                                                        ),
                                                    ],
                                                    final_expr: None,
                                                }),
                                            })),
                                            if_false: Some(Box::new(Expr::R(RExpr {
                                                ty: Type::Never,
                                                span: None,
                                                kind: RExprKind::Break(lowered_body.loop_id, None),
                                            }))),
                                        },
                                    })),
                                ),
                            }),
                        ],
                        final_expr: None,
                    }),
                }))
            }
            ast::Expr::ArrayInitializer(e) => {
                let length = e.elements.len() as u64;
                let expect_element_type = match expect_type {
                    Some(Type::Array {
                        element,
                        length: expected_length,
                    }) => {
                        if length != expected_length {
                            return Err(Error::new(format!(
                                "expected array of length {expected_length}, got {length}"
                            ))
                            .with_span(expr.span()));
                        }
                        Some(self.typesystem.get_type(element))
                    }
                    Some(other) => {
                        return Err(
                            Error::new(format!("expected expr of type {other:?}, got array initializer"))
                                .with_span(expr.span()),
                        );
                    }
                    None => None,
                };
                let lowered_elements = e
                    .elements
                    .iter()
                    .map(|expr| self.lower_expr(expr, expect_element_type))
                    .collect::<Result<Vec<_>, _>>()?;
                let element_ty = expect_element_type.unwrap_or_else(|| {
                    lowered_elements
                        .iter()
                        .map(|expr| expr.ty())
                        .find(|ty| *ty != Type::Never)
                        .unwrap_or(Type::Never)
                });
                let element_ty_id = self.typesystem.get_type_id(element_ty);
                let array_ty = Type::Array {
                    element: element_ty_id,
                    length,
                };
                for expr in &lowered_elements {
                    if expr.ty() != Type::Never && expr.ty() != element_ty {
                        return Err(Error::expr_type_missmatch(element_ty, expr.ty(), expr.span().unwrap()));
                    }
                }
                Ok(Expr::R(RExpr {
                    ty: array_ty,
                    span,
                    kind: RExprKind::ArrayInitializer(lowered_elements),
                }))
            }
            ast::Expr::StructInitializer(e) => {
                let (ty, sid) = match &e.struct_name {
                    Some(name) => {
                        let ty =
                            self.type_namespace.get(&name.value).copied().ok_or_else(|| {
                                Error::new(format!("unknown type {:?}", name.value)).with_span(name.span)
                            })?;
                        let sid = match ty {
                            Type::Struct(sid) => sid,
                            other => {
                                return Err(
                                    Error::new(format!("{} is not a struct type, but {other:?}", name.value))
                                        .with_span(name.span),
                                );
                            }
                        };
                        if let Some(expect_type) = expect_type
                            && expect_type != ty
                        {
                            return Err(Error::expr_type_missmatch(expect_type, ty, expr.span()));
                        }
                        (ty, sid)
                    }
                    None => match expect_type {
                        Some(ty @ Type::Struct(sid)) => (ty, sid),
                        None => return Err(Error::new("type annotations needed").with_span(expr.span())),
                        Some(other) => {
                            return Err(
                                Error::new(format!("expected expr of type {other:?}, got struct initializer"))
                                    .with_span(expr.span()),
                            );
                        }
                    },
                };
                let struct_def = self.typesystem.get_struct(sid);
                if let Some(missing_field) = struct_def
                    .fields
                    .iter()
                    .find(|f| !e.fields.iter().any(|ef| ef.name.value == f.name.value))
                {
                    return Err(
                        Error::new(format!("missing field: {}", missing_field.name.value)).with_span(expr.span())
                    );
                }
                let mut lowered_fields = Vec::new();
                for field in &e.fields {
                    let (_offset, f_ty) = self.typesystem.get_struct_field(sid, &field.name)?;
                    let expr = self.lower_expr(&field.value, Some(f_ty))?;
                    lowered_fields.push((field.name.value.clone(), expr));
                }
                Ok(Expr::R(RExpr {
                    ty,
                    span,
                    kind: RExprKind::StructInitializer(lowered_fields),
                }))
            }
            ast::Expr::Return(return_expr) => Ok(Expr::R(RExpr {
                ty: Type::Never,
                span,
                kind: RExprKind::Return(Box::new(
                    self.lower_expr(&return_expr.value, Some(self.decl.return_ty))?,
                )),
            })),
            ast::Expr::Break(break_expr) => {
                let loop_ctx = self.scope.loop_context().ok_or_else(|| {
                    Error::new("break expressions are only allowed inside loops")
                        .with_span(break_expr.break_keyword_span)
                })?;
                let expect_type = loop_ctx.expect_type;
                let break_from = loop_ctx.break_from;
                let lowered_value = break_expr
                    .value
                    .as_ref()
                    .map(|expr| self.lower_expr(expr, expect_type))
                    .transpose()?;
                self.scope.loop_context().unwrap().break_used_with_type =
                    Some(lowered_value.as_ref().map_or(Type::Void, |expr| expr.ty()));
                Ok(Expr::R(RExpr {
                    ty: Type::Never,
                    span,
                    kind: RExprKind::Break(break_from, lowered_value.map(Box::new)),
                }))
            }
            ast::Expr::Literal(literal_expr) => match &literal_expr.value {
                ast::LiteralExprValue::Number(number) => {
                    let int_ty = match expect_type {
                        None => IntType::I32,
                        Some(Type::Int(i)) => i,
                        Some(other) => {
                            return Err(Error::expr_type_missmatch(
                                other,
                                Type::Int(IntType::I32),
                                literal_expr.span,
                            ));
                        }
                    };
                    Ok(Expr::R(RExpr {
                        ty: Type::Int(int_ty),
                        span,
                        kind: RExprKind::ConstNumber(*number),
                    }))
                }
                ast::LiteralExprValue::String(string) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::I8_PTR
                    {
                        Err(Error::expr_type_missmatch(expect_type, Type::I8_PTR, literal_expr.span))
                    } else {
                        Ok(Expr::R(RExpr {
                            ty: Type::I8_PTR,
                            span,
                            kind: RExprKind::ConstString(string.clone()),
                        }))
                    }
                }
                ast::LiteralExprValue::Bool(bool) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        Err(Error::expr_type_missmatch(expect_type, Type::Bool, literal_expr.span))
                    } else {
                        Ok(Expr::R(RExpr {
                            ty: Type::Bool,
                            span,
                            kind: RExprKind::ConstBool(*bool),
                        }))
                    }
                }
                ast::LiteralExprValue::Undefined => {
                    let ty = expect_type
                        .ok_or_else(|| Error::new("type annotations needed").with_span(literal_expr.span))?;
                    Ok(Expr::R(RExpr {
                        ty,
                        span,
                        kind: RExprKind::Undefined,
                    }))
                }
            },
            ast::Expr::FunctionCallExpr(function_call_expr) => {
                let callee_id = self
                    .functions_namespace
                    .get(&function_call_expr.name.value)
                    .ok_or_else(|| {
                        Error::new(format!("function {:?} not found", function_call_expr.name.value))
                            .with_span(function_call_expr.name.span)
                    })?;
                let callee = &self.functions[&callee_id];
                if callee.is_variadic {
                    if callee.args.len() > function_call_expr.args.len() {
                        return Err(Error::new(format!(
                            "expected at least {} argument(s), found {}",
                            callee.args.len(),
                            function_call_expr.args.len()
                        ))
                        .with_span(function_call_expr.args_span));
                    }
                } else if callee.args.len() != function_call_expr.args.len() {
                    return Err(Error::new(format!(
                        "expected {} argument(s), found {}",
                        callee.args.len(),
                        function_call_expr.args.len()
                    ))
                    .with_span(function_call_expr.args_span));
                }
                let mut args_exprs = Vec::new();
                for (arg_i, arg_expr) in function_call_expr.args.iter().enumerate() {
                    let expect_arg_type = callee.args.get(arg_i).map(|a| a.1);
                    args_exprs.push(self.lower_expr(arg_expr, expect_arg_type)?);
                }
                if let Some(expect_type) = expect_type
                    && expect_type != callee.return_ty
                    && callee.return_ty != Type::Never
                {
                    return Err(Error::expr_type_missmatch(
                        expect_type,
                        callee.return_ty,
                        function_call_expr.span(),
                    ));
                }
                Ok(Expr::R(RExpr {
                    ty: callee.return_ty,
                    span,
                    kind: RExprKind::FunctionCall(*callee_id, args_exprs),
                }))
            }
            ast::Expr::Assignment(e) => {
                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::expr_type_missmatch(expect_type, Type::Void, expr.span()));
                }
                let lowered_place = self.lower_expr(&e.place, None)?.expect_lvalue()?;
                let lowered_value = self.lower_expr(&e.value, Some(lowered_place.ty))?;
                Ok(Expr::R(RExpr {
                    ty: Type::Void,
                    span,
                    kind: RExprKind::Store(Box::new(lowered_place), Box::new(lowered_value)),
                }))
            }
            ast::Expr::CompoundAssignment(e) => {
                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::expr_type_missmatch(expect_type, Type::Void, expr.span()));
                }
                let lowered_place = self.lower_expr(&e.place, None)?.expect_lvalue()?;
                let lowered_value = self.lower_expr(&e.value, Some(lowered_place.ty))?;
                let operands_ty = lowered_place.ty;
                if !operands_ty.is_int() {
                    return Err(Error::new(format!(
                        "arithemitc can only be performed on integers, not {operands_ty:?}"
                    ))
                    .with_span(e.op_span));
                }
                let place_ptr_ty = operands_ty.make_ptr(self.typesystem);
                let place_ptr_expr = Expr::R(RExpr {
                    ty: place_ptr_ty,
                    span: None,
                    kind: RExprKind::GetPointer(Box::new(lowered_place)),
                });
                let tmp_var_id = VariableId::new();
                Ok(Expr::R(RExpr {
                    ty: Type::Void,
                    span,
                    kind: RExprKind::Block(BlockExpr {
                        variables: vec![(tmp_var_id, place_ptr_ty)],
                        statements: vec![
                            Expr::set_var(tmp_var_id, place_ptr_expr),
                            Expr::R(RExpr {
                                ty: Type::Void,
                                span: None,
                                kind: RExprKind::Store(
                                    Box::new(LExpr::dereference(Expr::get_var(tmp_var_id, place_ptr_ty), operands_ty)),
                                    Box::new(Expr::R(RExpr {
                                        ty: operands_ty,
                                        span: None,
                                        kind: RExprKind::BinOp(
                                            BinaryOp::Arithmetic(e.op),
                                            Box::new(Expr::L(LExpr::dereference(
                                                Expr::get_var(tmp_var_id, place_ptr_ty),
                                                operands_ty,
                                            ))),
                                            Box::new(lowered_value),
                                        ),
                                    })),
                                ),
                            }),
                        ],
                        final_expr: None,
                    }),
                }))
            }
            ast::Expr::Binary(binary_expr) => match binary_expr.op {
                BinaryOp::Cmp(_) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        return Err(Error::expr_type_missmatch(expect_type, Type::Bool, expr.span()));
                    }
                    let lowered_lhs = self.lower_expr(&binary_expr.lhs, None)?;
                    let lowered_rhs = self.lower_expr(&binary_expr.rhs, Some(lowered_lhs.ty()))?;
                    let operands_ty = coalesce_types(lowered_lhs.ty(), lowered_rhs.ty());
                    if !operands_ty.is_int() {
                        return Err(
                            Error::new(format!("only integer types can be compared, not {operands_ty:?}"))
                                .with_span(binary_expr.op_span),
                        );
                    }
                    Ok(Expr::R(RExpr {
                        ty: Type::Bool,
                        span,
                        kind: RExprKind::BinOp(binary_expr.op, Box::new(lowered_lhs), Box::new(lowered_rhs)),
                    }))
                }
                BinaryOp::Arithmetic(_) => {
                    let lowered_lhs = self.lower_expr(&binary_expr.lhs, None)?;
                    let lowered_rhs = self.lower_expr(&binary_expr.rhs, Some(lowered_lhs.ty()))?;
                    let operands_ty = coalesce_types(lowered_lhs.ty(), lowered_rhs.ty());
                    if !operands_ty.is_int() {
                        return Err(Error::new(format!(
                            "arithemitc can only be performed on integers, not {operands_ty:?}"
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    if let Some(expect_type) = expect_type
                        && expect_type != operands_ty
                    {
                        return Err(Error::expr_type_missmatch(expect_type, operands_ty, expr.span()));
                    }
                    Ok(Expr::R(RExpr {
                        ty: operands_ty,
                        span,
                        kind: RExprKind::BinOp(binary_expr.op, Box::new(lowered_lhs), Box::new(lowered_rhs)),
                    }))
                }
                BinaryOp::LogicalOr => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        return Err(Error::expr_type_missmatch(expect_type, Type::Bool, expr.span()));
                    }
                    let lowered_lhs = self.lower_expr(&binary_expr.lhs, Some(Type::Bool))?;
                    let lowered_rhs = self.lower_expr(&binary_expr.rhs, Some(Type::Bool))?;
                    Ok(Expr::R(RExpr {
                        ty: Type::Bool,
                        span,
                        kind: RExprKind::If {
                            cond: Box::new(lowered_lhs),
                            if_true: Box::new(Expr::const_bool(true)),
                            if_false: Some(Box::new(lowered_rhs)),
                        },
                    }))
                }
                BinaryOp::LogicalAnd => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        return Err(Error::expr_type_missmatch(expect_type, Type::Bool, expr.span()));
                    }
                    let lowered_lhs = self.lower_expr(&binary_expr.lhs, Some(Type::Bool))?;
                    let lowered_rhs = self.lower_expr(&binary_expr.rhs, Some(Type::Bool))?;
                    Ok(Expr::R(RExpr {
                        ty: Type::Bool,
                        span,
                        kind: RExprKind::If {
                            cond: Box::new(lowered_lhs),
                            if_true: Box::new(lowered_rhs),
                            if_false: Some(Box::new(Expr::const_bool(false))),
                        },
                    }))
                }
            },
            ast::Expr::Unary(unary_expr) => match unary_expr.op {
                ast::UnaryOp::Negate => {
                    let lowered_rhs = self.lower_expr(&unary_expr.rhs, None)?;
                    let int_ty = match lowered_rhs.ty() {
                        Type::Int(i) if i.is_signed() => i,
                        other => {
                            return Err(Error::new(format!("only signed integer can be negated, not {other:?}"))
                                .with_span(unary_expr.op_span));
                        }
                    };
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Int(int_ty)
                    {
                        return Err(Error::expr_type_missmatch(expect_type, Type::Int(int_ty), expr.span()));
                    }
                    Ok(Expr::R(RExpr {
                        ty: Type::Int(int_ty),
                        span,
                        kind: RExprKind::BinOp(
                            BinaryOp::Arithmetic(ArithmeticOp::Sub),
                            Box::new(Expr::const_number(0, Type::Int(int_ty))),
                            Box::new(lowered_rhs),
                        ),
                    }))
                }
                ast::UnaryOp::Not => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        return Err(Error::expr_type_missmatch(expect_type, Type::Bool, expr.span()));
                    }
                    let lowered_rhs = self.lower_expr(&unary_expr.rhs, Some(Type::Bool))?;
                    Ok(Expr::R(RExpr {
                        ty: Type::Bool,
                        span,
                        kind: RExprKind::Not(Box::new(lowered_rhs)),
                    }))
                }
                ast::UnaryOp::AddressOf => {
                    let lowered_rhs = self.lower_expr(&unary_expr.rhs, None)?.expect_lvalue()?;
                    let ty = lowered_rhs.ty.make_ptr(self.typesystem);
                    if let Some(expect_type) = expect_type
                        && expect_type != ty
                    {
                        return Err(Error::expr_type_missmatch(expect_type, ty, expr.span()));
                    }
                    Ok(Expr::R(RExpr {
                        ty,
                        span,
                        kind: RExprKind::GetPointer(Box::new(lowered_rhs)),
                    }))
                }
            },
            ast::Expr::AsCast(as_cast_expr) => {
                let ty = self.typesystem.type_from_ast(self.type_namespace, &as_cast_expr.ty)?;
                if let Some(expect_type) = expect_type
                    && expect_type != ty
                {
                    return Err(Error::expr_type_missmatch(expect_type, ty, expr.span()));
                }
                let lowered_expr = self.lower_expr(&as_cast_expr.expr, None)?; // TODO: pass ty as a hint (but not a requirement!)
                match (lowered_expr.ty(), ty) {
                    (Type::Int(_), Type::Int(_)) | (Type::Ptr { .. }, Type::Ptr { .. }) => Ok(Expr::R(RExpr {
                        ty,
                        span,
                        kind: RExprKind::Cast(Box::new(lowered_expr)),
                    })),
                    (from, to) => {
                        Err(Error::new(format!("cannot cast from {from:?} to {to:?}")).with_span(as_cast_expr.as_span))
                    }
                }
            }
            ast::Expr::Comptime(_) => {
                unimplemented!("comptime blocks are not yet implemented")
            }
            ast::Expr::Range(_range_expr) => {
                unimplemented!("range expressions cannot be evaluated yet")
            }

            // Places
            //
            ast::Expr::Ident(ident) => {
                let (var_id, ty) = self
                    .scope
                    .lookup_variable(&ident.value)
                    .ok_or_else(|| Error::new(format!("variable {:?} not found", ident.value)).with_span(ident.span))?;
                Ok(Expr::L(LExpr {
                    ty,
                    span,
                    kind: LExprKind::Variable(var_id),
                }))
            }
            ast::Expr::FieldAccess(e) => {
                let lowered_lhs = self.lower_expr(&e.lhs, None)?;
                let struct_id = match lowered_lhs.ty() {
                    Type::Struct(struct_id) => struct_id,
                    _ => return Err(Error::new(format!("only structs have fields")).with_span(e.dot_span)),
                };
                let struct_ty = self.typesystem.get_struct(struct_id);
                let field = struct_ty
                    .fields
                    .iter()
                    .find(|f| f.name.value == e.field.value)
                    .ok_or_else(|| {
                        Error::new(format!(
                            "struct {:?} has no field {:?}",
                            struct_ty.name.value, e.field.value
                        ))
                    })?;
                Ok(match lowered_lhs {
                    Expr::R(lowered_lhs) => Expr::R(RExpr {
                        ty: field.ty,
                        span,
                        kind: RExprKind::Field(Box::new(lowered_lhs), field.name.value.clone()),
                    }),
                    Expr::L(lowered_lhs) => Expr::L(LExpr {
                        ty: field.ty,
                        span,
                        kind: LExprKind::Field(Box::new(lowered_lhs), field.name.value.clone()),
                    }),
                })
            }
            ast::Expr::Dereference(e) => {
                let expect_ptr_ty = expect_type.map(|ty| ty.make_ptr(self.typesystem));
                let lowered_ptr = self.lower_expr(&e.ptr, expect_ptr_ty)?;
                let ty = match lowered_ptr.ty() {
                    Type::Ptr { pointee } => pointee
                        .map(|id| self.typesystem.get_type(id))
                        .ok_or_else(|| Error::new("cannot dereference an opaque pointer").with_span(expr.span()))?,
                    other => {
                        return Err(
                            Error::new(format!("expected an expression of type pointer, got {other:?}"))
                                .with_span(e.ptr.span()),
                        );
                    }
                };
                Ok(Expr::L(LExpr {
                    ty,
                    span,
                    kind: LExprKind::Dereference(Box::new(lowered_ptr)),
                }))
            }
            ast::Expr::Index(e) => {
                let lowered_lhs = self.lower_expr(&e.lhs, None)?;
                let element_ty = match lowered_lhs.ty() {
                    Type::Array { element, length: _ } => self.typesystem.get_type(element),
                    other => {
                        return Err(Error::new(format!("expected an array, got {:?}", other)).with_span(e.lhs.span()));
                    }
                };
                let lowered_index = self.lower_expr(&e.index, Some(Type::Int(IntType::I32)))?;
                Ok(match lowered_lhs {
                    Expr::R(lowered_lhs) => Expr::R(RExpr {
                        ty: element_ty,
                        span,
                        kind: RExprKind::ArrayElement(Box::new(lowered_lhs), Box::new(lowered_index)),
                    }),
                    Expr::L(lowered_lhs) => Expr::L(LExpr {
                        ty: element_ty,
                        span,
                        kind: LExprKind::ArrayElement(Box::new(lowered_lhs), Box::new(lowered_index)),
                    }),
                })
            }
        }
    }

    /// Evaluate a block expression
    fn lower_block_expr(&mut self, expr: &ast::BlockExpr, expect_type: Option<Type>) -> Result<Expr, Error> {
        self.scope.push();

        let mut variables = Vec::new();
        let mut statements = Vec::new();

        for stmt in &expr.statements {
            match stmt {
                ast::Statement::Let(let_statement) => match let_statement {
                    ast::LetStatement::WithValue { name, ty, value } => {
                        let var_id = VariableId::new();
                        let var_ty = ty
                            .as_ref()
                            .map(|ty| self.typesystem.type_from_ast(self.type_namespace, ty))
                            .transpose()?;
                        let value_eval = self.lower_expr(value, var_ty)?;
                        variables.push((var_id, value_eval.ty()));
                        self.scope
                            .variables
                            .insert(name.value.clone(), (var_id, value_eval.ty()));
                        statements.push(Expr::set_var(var_id, value_eval));
                    }
                    ast::LetStatement::WithoutValue { name, ty } => {
                        let id = VariableId::new();
                        let ty = self.typesystem.type_from_ast(self.type_namespace, ty)?;
                        variables.push((id, ty));
                        self.scope.variables.insert(name.value.clone(), (id, ty));
                    }
                },
                ast::Statement::Expr(expr) => {
                    statements.push(self.lower_expr(expr, None)?);
                }
            }
        }

        let final_expr = expr
            .final_expr
            .as_ref()
            .map(|expr| self.lower_expr(expr, expect_type))
            .transpose()?;

        self.scope.pop();

        if final_expr.is_none()
            && let Some(expect_type) = expect_type
            && expect_type != Type::Void
        {
            return Err(
                Error::new(format!("expected expr of type {expect_type:?}, found end-of-block"))
                    .with_span(expr.closing_brace_span),
            );
        }

        Ok(Expr::R(RExpr {
            ty: final_expr.as_ref().map_or(Type::Void, |expr| expr.ty()),
            span: Some(expr.span()),
            kind: RExprKind::Block(BlockExpr {
                variables,
                statements,
                final_expr: final_expr.map(Box::new),
            }),
        }))
    }

    fn lower_loop_body(
        &mut self,
        body: &ast::BlockExpr,
        expect_type: Option<Type>,
    ) -> Result<LowerLoopBodyResult, Error> {
        let loop_id = LoopId::new();
        self.scope.push();
        self.scope.loop_context = Some(LoopContext {
            break_from: loop_id,
            break_used_with_type: None,
            expect_type,
        });
        let body = self.lower_block_expr(body, Some(Type::Void))?;
        let loop_ctx = self.scope.loop_context.unwrap();
        self.scope.pop();
        Ok(LowerLoopBodyResult {
            loop_id,
            body,
            break_used_with_type: loop_ctx.break_used_with_type,
        })
    }
}
