use std::collections::HashMap;

use super::*;
use crate::ast;
use crate::common::{ArithmeticOp, CmpOp};

pub fn lower_function_decl<'ctx>(
    ctx: Context<'ctx>,
    ast: &ast::Function,
    annotations: &[ast::Annotation],
    types_scope: &mut TypesScope<'ctx>,
) -> Result<Function<'ctx>, Error> {
    let id = FunctionId::new();

    let mut is_pure = false;
    let mut is_intrinsic = false;
    for annotation in annotations {
        match annotation.ident.value.as_str() {
            "pure" => is_pure = true,
            "intrinsic" => is_intrinsic = true,
            _ => return Err(Error::unknown_annotation(annotation)),
        }
    }

    types_scope.push();

    types_scope.insert_ast_type_parameters(ctx, TypeParameterOwner::Function(id), &ast.type_parameters)?;

    let mut args: Vec<(String, Type)> = Vec::new();
    for arg in &ast.args {
        if args.iter().any(|x| x.0 == arg.name.value) {
            return Err(Error::new("argument with this name already exists").with_span(arg.name.span));
        }
        args.push((arg.name.value.clone(), type_from_ast(ctx, types_scope, &arg.ty)?));
    }

    let return_ty = ast
        .return_ty
        .as_ref()
        .map(|ty| type_from_ast(ctx, types_scope, ty))
        .transpose()?
        .unwrap_or_else(|| ctx.types().unit);

    types_scope.pop();

    Ok(Function {
        id,
        name_ident: ast.name.clone(),
        mangled_name: ast.name.value.clone(),
        debug_name: ast.name.value.clone(),
        monomorphized_for: None,
        type_parameters: ast.type_parameters.len(),
        args,
        return_ty,
        is_variadic: ast.is_variadic,
        is_pure,
        is_intrinsic,
        body: None,
    })
}

/// Construct an IR of a function from its AST
pub fn lower_function_body<'ctx>(
    ctx: Context<'ctx>,
    decl: &Function<'ctx>,
    ast: &ast::Function,
    body: &ast::BlockExpr,
    functions_namespace: &HashMap<String, FunctionId>,
    functions: &BTreeMap<FunctionId, Function<'ctx>>,
    types_scope: &mut TypesScope<'ctx>,
) -> Result<Expr<'ctx>, Error> {
    if decl.is_variadic {
        return Err(Error::new("defining variadic functions is not supported").with_span(decl.name_ident.span));
    }

    let mut builder = FunctionLoweringCtx::new(ctx, decl, functions_namespace, functions, types_scope);

    let mut variables = Vec::new();
    let mut exprs = Vec::new();

    for (arg_i, (arg_name, arg_ty)) in decl.args.iter().enumerate() {
        let arg_var_id = VariableId::new();
        builder.scope.variables.insert(arg_name.clone(), (arg_var_id, *arg_ty));
        variables.push(VariableDeclaration {
            id: arg_var_id,
            ty: *arg_ty,
            debug_name: arg_name.clone(),
        });
        exprs.push(Expr::set_var(
            ctx,
            arg_var_id,
            Expr {
                ty: *arg_ty,
                span: None,
                kind: ExprKind::Argument(arg_i),
            },
        ));
    }

    builder.types_scope.push();
    builder
        .types_scope
        .insert_ast_type_parameters(ctx, TypeParameterOwner::Function(decl.id), &ast.type_parameters)?;
    exprs.push(builder.lower_block_expr(body, Some(decl.return_ty))?);
    builder.types_scope.pop();

    Ok(Expr {
        ty: decl.return_ty,
        span: Some(body.span()),
        kind: ExprKind::Block(BlockExpr { variables, exprs }),
    })
}

/// A function's AST -> IR_TREE lowering context
struct FunctionLoweringCtx<'a, 'ctx> {
    ctx: Context<'ctx>,
    decl: &'a Function<'ctx>,
    functions_namespace: &'a HashMap<String, FunctionId>,
    functions: &'a BTreeMap<FunctionId, Function<'ctx>>,
    types_scope: &'a mut TypesScope<'ctx>,
    scope: Scope<'ctx>,
}

/// A lexical scope
#[derive(Default)]
struct Scope<'ctx> {
    variables: HashMap<String, (VariableId, Type<'ctx>)>,
    loop_context: Option<LoopContext<'ctx>>,
    parent: Option<Box<Self>>,
}

#[derive(Clone, Copy)]
struct LoopContext<'ctx> {
    loop_id: LoopId,
    break_used_with_type: Option<Type<'ctx>>,
    expect_type: Option<Type<'ctx>>,
}

impl<'ctx> Scope<'ctx> {
    /// Create a nested scope
    fn push(&mut self) {
        let parent = std::mem::take(self);
        self.parent = Some(Box::new(parent));
    }

    /// Pop the latest nested scope
    fn pop(&mut self) {
        *self = *self.parent.take().unwrap();
    }

    /// Lookup a variable by its name, recursively traversing the list of scopes
    fn lookup_variable(&self, name: &str) -> Option<(VariableId, Type<'ctx>)> {
        if let Some(definition_id) = self.variables.get(name) {
            return Some(*definition_id);
        }
        if let Some(parent) = &self.parent {
            return parent.lookup_variable(name);
        }
        None
    }

    /// Recursively lookup a loop context
    fn loop_context(&mut self) -> Option<&mut LoopContext<'ctx>> {
        if let Some(ctx) = &mut self.loop_context {
            return Some(ctx);
        }
        if let Some(parent) = &mut self.parent {
            return parent.loop_context();
        }
        None
    }
}

struct LowerLoopBodyResult<'ctx> {
    loop_id: LoopId,
    body: Expr<'ctx>,
    break_used_with_type: Option<Type<'ctx>>,
}

impl<'a, 'ctx> FunctionLoweringCtx<'a, 'ctx> {
    /// Create a new function lowering context
    fn new(
        ctx: Context<'ctx>,
        decl: &'a Function<'ctx>,
        functions_namespace: &'a HashMap<String, FunctionId>,
        functions: &'a BTreeMap<FunctionId, Function<'ctx>>,
        types_scope: &'a mut TypesScope<'ctx>,
    ) -> Self {
        Self {
            ctx,
            decl,
            functions_namespace,
            functions,
            types_scope,
            scope: Scope::default(),
        }
    }

    /// lower an expression
    fn lower_expr(&mut self, expr: &ast::Expr, expect_type: Option<Type<'ctx>>) -> Result<Expr<'ctx>, Error> {
        let span = Some(expr.span);
        match &expr.kind {
            ast::ExprKind::Ident(ident) => {
                let (var_id, ty) = self
                    .scope
                    .lookup_variable(&ident.value)
                    .ok_or_else(|| Error::new(format!("variable {:?} not found", ident.value)).with_span(ident.span))?;
                if let Some(expect_type) = expect_type
                    && expect_type != ty
                {
                    return Err(Error::expr_type_mismatch(expect_type, ty, expr.span));
                }
                Ok(Expr {
                    ty,
                    span,
                    kind: ExprKind::Load(Place {
                        ty,
                        span,
                        kind: PlaceKind::Variable(var_id),
                    }),
                })
            }
            ast::ExprKind::Block(block_expr) => self.lower_block_expr(block_expr, expect_type),
            ast::ExprKind::If(cond, if_true, if_false) => {
                let expected_non_unit = expect_type.filter(|ty| !ty.is_unit());

                let expect_type = match (expected_non_unit, if_false.is_some()) {
                    (None, false) => Some(self.ctx.types().unit),

                    (_, false) => {
                        return Err(Error::new(format!(
                            "if expression expected to evaluate to type {expect_type:?}, so it must have an else branch"
                        ))
                        .with_span(expr.span));
                    }

                    (expect_type, true) => expect_type,
                };

                let lowered_cond = self.lower_expr(cond, Some(self.ctx.types().bool))?;

                let lowered_if_true = self.lower_block_expr(if_true, expect_type)?;

                let expect_type = if lowered_if_true.ty.is_never() {
                    expect_type
                } else {
                    Some(lowered_if_true.ty)
                };

                let lowered_if_false = if_false
                    .as_ref()
                    .map(|expr| self.lower_expr(expr, expect_type))
                    .transpose()?
                    .unwrap_or_else(|| Expr::unit(self.ctx));

                Ok(Expr {
                    ty: coalesce_types(lowered_if_true.ty, lowered_if_false.ty),
                    span,
                    kind: ExprKind::If {
                        cond: Box::new(lowered_cond),
                        if_true: Box::new(lowered_if_true),
                        if_false: Box::new(lowered_if_false),
                    },
                })
            }
            ast::ExprKind::Loop(body) => {
                let lowered_body = self.lower_loop_body(body, expect_type)?;
                Ok(Expr {
                    ty: lowered_body
                        .break_used_with_type
                        .unwrap_or_else(|| self.ctx.types().never),
                    span,
                    kind: ExprKind::Loop(lowered_body.loop_id, Box::new(lowered_body.body)),
                })
            }
            ast::ExprKind::While(cond, body) => {
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
                    && !expect_type.is_unit()
                {
                    return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().unit, expr.span));
                }
                let lowered_cond = self.lower_expr(cond, Some(self.ctx.types().bool))?;
                let lowered_body = self.lower_loop_body(body, Some(self.ctx.types().unit))?;
                Ok(Expr {
                    ty: self.ctx.types().unit,
                    span,
                    kind: ExprKind::Loop(
                        lowered_body.loop_id,
                        Box::new(Expr {
                            ty: self.ctx.types().unit,
                            span,
                            kind: ExprKind::If {
                                cond: Box::new(lowered_cond),
                                if_true: Box::new(lowered_body.body),
                                if_false: Box::new(Expr {
                                    ty: self.ctx.types().never,
                                    span: None,
                                    kind: ExprKind::Break(lowered_body.loop_id, Box::new(Expr::unit(self.ctx))),
                                }),
                            },
                        }),
                    ),
                })
            }
            ast::ExprKind::For(i, i_ty, iterator_range, body) => {
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
                //             $var += 1
                //             { $body }
                //         } else {
                //             break
                //         }
                //     }
                // }

                if let Some(expect_type) = expect_type
                    && !expect_type.is_unit()
                {
                    return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().unit, expr.span));
                }

                let i_ty = i_ty
                    .as_ref()
                    .map(|ast| type_from_ast(self.ctx, self.types_scope, ast))
                    .transpose()?;

                let ast::ExprKind::Range(range_from, range_to) = &iterator_range.kind else {
                    return Err(
                        Error::new("only range exprs (e.g. 'a..b') are supported as iterator in 'for' yet")
                            .with_span(iterator_range.span),
                    );
                };

                let lowered_range_from = self.lower_expr(range_from, i_ty)?;
                let var_type = lowered_range_from.ty;
                let var_type_int = var_type
                    .as_int()
                    .ok_or_else(|| Error::new("expected an expression of integer type").with_span(range_from.span))?;
                let lowered_range_to = self.lower_expr(range_to, Some(var_type))?;

                let var_id = VariableId::new();
                let target_id = VariableId::new();
                let shadowed_var_id = VariableId::new();

                self.scope.push();
                self.scope
                    .variables
                    .insert(i.value.clone(), (shadowed_var_id, var_type));
                let lowered_body = self.lower_loop_body(body, Some(self.ctx.types().unit))?;
                self.scope.pop();

                Ok(Expr {
                    ty: self.ctx.types().unit,
                    span,
                    kind: ExprKind::Block(BlockExpr {
                        variables: vec![
                            VariableDeclaration {
                                id: var_id,
                                ty: var_type,
                                debug_name: format!("<immutable-{}>", i.value),
                            },
                            VariableDeclaration {
                                id: target_id,
                                ty: var_type,
                                debug_name: format!("<tmp-{}-target>", i.value),
                            },
                            VariableDeclaration {
                                id: shadowed_var_id,
                                ty: var_type,
                                debug_name: i.value.clone(),
                            },
                        ],
                        exprs: vec![
                            Expr::set_var(self.ctx, var_id, lowered_range_from),
                            Expr::set_var(self.ctx, target_id, lowered_range_to),
                            Expr {
                                ty: self.ctx.types().unit,
                                span: None,
                                kind: ExprKind::Loop(
                                    lowered_body.loop_id,
                                    Box::new(Expr {
                                        ty: self.ctx.types().unit,
                                        span: None,
                                        kind: ExprKind::If {
                                            cond: Box::new(Expr {
                                                ty: self.ctx.types().bool,
                                                span: None,
                                                kind: ExprKind::Cmp(
                                                    CmpOp::Less,
                                                    Box::new(Expr::get_var(var_id, var_type)),
                                                    Box::new(Expr::get_var(target_id, var_type)),
                                                ),
                                            }),
                                            if_true: Box::new(Expr {
                                                ty: self.ctx.types().unit,
                                                span: None,
                                                kind: ExprKind::Block(BlockExpr {
                                                    variables: Vec::new(),
                                                    exprs: vec![
                                                        Expr::set_var(
                                                            self.ctx,
                                                            shadowed_var_id,
                                                            Expr::get_var(var_id, var_type),
                                                        ),
                                                        Expr {
                                                            ty: self.ctx.types().unit,
                                                            span: None,
                                                            kind: ExprKind::InPlaceArithmetic(
                                                                ArithmeticOp::Add,
                                                                Place::var(var_id, var_type),
                                                                Box::new(Expr::new_const(
                                                                    self.ctx,
                                                                    Constant::int(self.ctx, 1, var_type_int).unwrap(),
                                                                )),
                                                            ),
                                                        },
                                                        lowered_body.body,
                                                    ],
                                                }),
                                            }),
                                            if_false: Box::new(Expr {
                                                ty: self.ctx.types().never,
                                                span: None,
                                                kind: ExprKind::Break(
                                                    lowered_body.loop_id,
                                                    Box::new(Expr::unit(self.ctx)),
                                                ),
                                            }),
                                        },
                                    }),
                                ),
                            },
                        ],
                    }),
                })
            }
            ast::ExprKind::ArrayInitializer(elements) => {
                let length = elements.len() as u64;
                let expect_element_type = match expect_type {
                    Some(expect_type) => match expect_type.as_array() {
                        Some((element_ty, expected_length)) => {
                            if length != expected_length {
                                return Err(Error::new(format!(
                                    "expected array of length {expected_length}, got {length}"
                                ))
                                .with_span(expr.span));
                            }
                            Some(element_ty)
                        }
                        None => {
                            return Err(Error::new(format!(
                                "expected expr of type {expect_type:?}, got array initializer"
                            ))
                            .with_span(expr.span));
                        }
                    },
                    None => None,
                };
                let lowered_elements = elements
                    .iter()
                    .map(|expr| self.lower_expr(expr, expect_element_type))
                    .collect::<Result<Vec<_>, _>>()?;
                let element_ty = expect_element_type.unwrap_or_else(|| {
                    lowered_elements
                        .iter()
                        .map(|expr| expr.ty)
                        .find(|ty| !ty.is_never())
                        .unwrap_or_else(|| self.ctx.types().never)
                });
                for expr in &lowered_elements {
                    if !expr.ty.is_never() && expr.ty != element_ty {
                        return Err(Error::expr_type_mismatch(element_ty, expr.ty, expr.span.unwrap()));
                    }
                }
                Ok(Expr {
                    ty: Type::new(self.ctx, TypeInfo::Array { element_ty, length }),
                    span,
                    kind: ExprKind::ArrayInitializer(lowered_elements),
                })
            }
            ast::ExprKind::StructInitializer(fields) => {
                let ty = expect_type.ok_or_else(|| Error::new("type annotations needed").with_span(expr.span))?;
                let (struct_, type_arguments) = ty.as_struct().ok_or_else(|| {
                    Error::new(format!("expected expr of type {}, got struct initializer", ty.render()))
                        .with_span(expr.span)
                })?;
                if let Some(missing_field) = struct_
                    .info()
                    .fields
                    .iter()
                    .find(|f| !fields.iter().any(|ef| ef.name.value == f.0.value))
                {
                    return Err(Error::new(format!("missing field: {}", missing_field.0.value)).with_span(expr.span));
                }
                let mut lowered_fields: Vec<(StructFieldId, Expr)> = Vec::new();
                for field in fields {
                    let (_, field_id) = struct_
                        .info()
                        .fields
                        .iter()
                        .find(|(name, _)| name.value == field.name.value)
                        .ok_or_else(|| {
                            Error::new(format!("struct {struct_:?} has no field {}", field.name.value))
                                .with_span(field.name.span)
                        })?;
                    if lowered_fields.iter().any(|x| x.0 == *field_id) {
                        return Err(Error::new("duplicate field entry").with_span(field.name.span));
                    }
                    let field_ty = self.ctx.type_of_struct_field(*field_id, type_arguments);
                    let expr = self.lower_expr(&field.value, Some(field_ty))?;
                    lowered_fields.push((*field_id, expr));
                }
                Ok(Expr {
                    ty,
                    span,
                    kind: ExprKind::StructInitializer(lowered_fields),
                })
            }
            ast::ExprKind::Return(value) => {
                if value.is_none() && !self.decl.return_ty.is_unit() {
                    return Err(
                        Error::new(format!("a return value of type {:?} is expected", self.decl.return_ty))
                            .with_span(expr.span),
                    );
                }
                let lowered_value = value
                    .as_ref()
                    .map(|expr| self.lower_expr(expr, Some(self.decl.return_ty)))
                    .transpose()?
                    .unwrap_or_else(|| Expr::unit(self.ctx));
                Ok(Expr {
                    ty: self.ctx.types().never,
                    span,
                    kind: ExprKind::Return(Box::new(lowered_value)),
                })
            }
            ast::ExprKind::Break(value) => {
                let loop_ctx = self.scope.loop_context().ok_or_else(|| {
                    Error::new("'break' expressions are only allowed inside loops").with_span(expr.span)
                })?;
                if value.is_none()
                    && let Some(expect_type) = loop_ctx.expect_type
                    && !expect_type.is_unit()
                {
                    return Err(
                        Error::new(format!("a break value of type {expect_type:?} is expected")).with_span(expr.span)
                    );
                }
                let expect_type = loop_ctx.expect_type;
                let loop_id = loop_ctx.loop_id;
                let lowered_value = value
                    .as_ref()
                    .map(|expr| self.lower_expr(expr, expect_type))
                    .transpose()?
                    .unwrap_or_else(|| Expr::unit(self.ctx));
                if !lowered_value.ty.is_never() {
                    let loop_ctx = self.scope.loop_context().unwrap();
                    loop_ctx.break_used_with_type = Some(lowered_value.ty);
                    loop_ctx.expect_type = Some(lowered_value.ty);
                }
                Ok(Expr {
                    ty: self.ctx.types().never,
                    span,
                    kind: ExprKind::Break(loop_id, Box::new(lowered_value)),
                })
            }
            ast::ExprKind::Continue => {
                let loop_ctx = self.scope.loop_context().ok_or_else(|| {
                    Error::new("'continue' expressions are only allowed inside loops").with_span(expr.span)
                })?;
                Ok(Expr {
                    ty: self.ctx.types().never,
                    span,
                    kind: ExprKind::Continue(loop_ctx.loop_id),
                })
            }
            ast::ExprKind::Literal(literal) => match literal {
                ast::Literal::Number(number) => {
                    let ty = expect_type.unwrap_or_else(|| self.ctx.types().i32);
                    let int_ty = ty
                        .as_int()
                        .ok_or_else(|| Error::expr_type_mismatch(ty, self.ctx.types().i32, expr.span))?;
                    let Some(const_value) = Constant::int(self.ctx, *number, int_ty) else {
                        return Err(Error::new(format!("number does not fit into {int_ty:?}")).with_span(expr.span));
                    };
                    Ok(Expr {
                        ty,
                        span,
                        kind: ExprKind::Const(const_value),
                    })
                }
                ast::Literal::String(string) => {
                    let ty = self.ctx.types().i8_ptr;
                    if let Some(expect_type) = expect_type
                        && expect_type != ty
                    {
                        Err(Error::expr_type_mismatch(expect_type, ty, expr.span))
                    } else {
                        Ok(Expr {
                            ty,
                            span,
                            kind: ExprKind::ConstString(string.clone()),
                        })
                    }
                }
                ast::Literal::Bool(bool) => {
                    let ty = self.ctx.types().bool;
                    if let Some(expect_type) = expect_type
                        && expect_type != ty
                    {
                        Err(Error::expr_type_mismatch(expect_type, ty, expr.span))
                    } else {
                        Ok(Expr {
                            ty,
                            span,
                            kind: ExprKind::Const(Constant::Bool(*bool)),
                        })
                    }
                }
                ast::Literal::Undefined => {
                    let ty = expect_type.ok_or_else(|| Error::new("type annotations needed").with_span(expr.span))?;
                    Ok(Expr {
                        ty,
                        span,
                        kind: ExprKind::Const(Constant::Undefined(ty)),
                    })
                }
                ast::Literal::Null => {
                    let pointee = match expect_type {
                        None => None,
                        Some(expect_type) => match expect_type.info() {
                            TypeInfo::Ptr { pointee } => *pointee,
                            _ => {
                                return Err(Error::expr_type_mismatch(
                                    expect_type,
                                    self.ctx.types().opaque_ptr,
                                    expr.span,
                                ));
                            }
                        },
                    };
                    let ty = Type::new(self.ctx, TypeInfo::Ptr { pointee });
                    Ok(Expr {
                        ty,
                        span,
                        kind: ExprKind::Const(Constant::Null(ty)),
                    })
                }
            },
            ast::ExprKind::FunctionCallExpr(name, type_arguments, args) => {
                let callee_id = self
                    .functions_namespace
                    .get(&name.value)
                    .ok_or_else(|| Error::new(format!("function {:?} not found", name.value)).with_span(name.span))?;
                let callee = &self.functions[callee_id];

                let type_arguments = TypeArguments::new(
                    self.ctx,
                    &type_arguments
                        .as_ref()
                        .map(|args| {
                            args.arguments
                                .iter()
                                .map(|arg| type_from_ast(self.ctx, self.types_scope, arg))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()?
                        .unwrap_or(Vec::new()),
                );
                if callee.type_parameters != type_arguments.0.get().len() {
                    return Err(Error::new(format!(
                        "expected {} type argument(s), found {}",
                        callee.type_parameters,
                        type_arguments.0.get().len(),
                    ))
                    .with_span(expr.span));
                }
                if callee.is_variadic {
                    if callee.args.len() > args.len() {
                        return Err(Error::new(format!(
                            "expected at least {} argument(s), found {}",
                            callee.args.len(),
                            args.len()
                        ))
                        .with_span(expr.span));
                    }
                } else if callee.args.len() != args.len() {
                    return Err(Error::new(format!(
                        "expected {} argument(s), found {}",
                        callee.args.len(),
                        args.len()
                    ))
                    .with_span(expr.span));
                }
                let mut lowered_args = Vec::new();
                for (arg_i, arg_expr) in args.iter().enumerate() {
                    let mut arg_type = callee.args.get(arg_i).map(|a| a.1);
                    if !type_arguments.is_empty()
                        && let Some(arg_type_generic) = arg_type
                    {
                        arg_type = Some(arg_type_generic.instantiate(
                            self.ctx,
                            TypeParameterOwner::Function(*callee_id),
                            type_arguments,
                        ));
                    }
                    lowered_args.push(self.lower_expr(arg_expr, arg_type)?);
                }

                let mut return_ty = callee.return_ty;
                if !type_arguments.is_empty() {
                    return_ty =
                        return_ty.instantiate(self.ctx, TypeParameterOwner::Function(*callee_id), type_arguments);
                }

                if let Some(expect_type) = expect_type
                    && expect_type != return_ty
                    && !return_ty.is_never()
                {
                    return Err(Error::expr_type_mismatch(expect_type, callee.return_ty, expr.span));
                }

                Ok(Expr {
                    ty: return_ty,
                    span,
                    kind: ExprKind::FunctionCall(*callee_id, type_arguments, lowered_args),
                })
            }
            ast::ExprKind::Assignment(place, value) => {
                if let Some(expect_type) = expect_type
                    && !expect_type.is_unit()
                {
                    return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().unit, expr.span));
                }
                let lowered_place = self.lower_expr(place, None)?.expect_place()?;
                let lowered_value = self.lower_expr(value, Some(lowered_place.ty))?;
                Ok(Expr {
                    ty: self.ctx.types().unit,
                    span,
                    kind: ExprKind::Store(lowered_place, Box::new(lowered_value)),
                })
            }
            ast::ExprKind::CompoundAssignment(place, op, value) => {
                if let Some(expect_type) = expect_type
                    && !expect_type.is_unit()
                {
                    return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().unit, expr.span));
                }
                let lowered_place = self.lower_expr(place, None)?.expect_place()?;
                let lowered_value = self.lower_expr(value, Some(lowered_place.ty))?;
                let operands_ty = lowered_place.ty;
                if !operands_ty.is_int() {
                    return Err(Error::new(format!(
                        "arithmetic can only be performed on integers, not {operands_ty:?}"
                    ))
                    .with_span(expr.span));
                }
                Ok(Expr {
                    ty: self.ctx.types().unit,
                    span,
                    kind: ExprKind::InPlaceArithmetic(*op, lowered_place, Box::new(lowered_value)),
                })
            }
            ast::ExprKind::Binary(lhs, op, rhs) => match *op {
                BinaryOp::Cmp(cmp_op) => {
                    if let Some(expect_type) = expect_type
                        && !expect_type.is_bool()
                    {
                        return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().bool, expr.span));
                    }
                    let lowered_lhs = self.lower_expr(lhs, None)?;
                    let lowered_rhs = self.lower_expr(rhs, Some(lowered_lhs.ty))?;
                    let operands_ty = coalesce_types(lowered_lhs.ty, lowered_rhs.ty);
                    match cmp_op {
                        _ if operands_ty.is_int() || operands_ty.is_ptr() => (),
                        CmpOp::Equal | CmpOp::NotEqual if operands_ty.is_bool() => (),
                        _ => {
                            return Err(Error::new(format!("cannot compare {operands_ty:?} with {cmp_op:?}"))
                                .with_span(expr.span));
                        }
                    }
                    Ok(Expr {
                        ty: self.ctx.types().bool,
                        span,
                        kind: ExprKind::Cmp(cmp_op, Box::new(lowered_lhs), Box::new(lowered_rhs)),
                    })
                }
                BinaryOp::Arithmetic(arithmetic_op) => {
                    let lowered_lhs = self.lower_expr(lhs, expect_type)?;
                    let lowered_rhs = self.lower_expr(
                        rhs,
                        Some(coalesce_types(lowered_lhs.ty, expect_type.unwrap_or(lowered_lhs.ty))),
                    )?;
                    let operands_ty = coalesce_types(lowered_lhs.ty, lowered_rhs.ty);
                    if !operands_ty.is_int() {
                        return Err(Error::new(format!(
                            "arithmetic can only be performed on integers, not {operands_ty:?}"
                        ))
                        .with_span(expr.span));
                    }
                    if let Some(expect_type) = expect_type
                        && expect_type != operands_ty
                    {
                        return Err(Error::expr_type_mismatch(expect_type, operands_ty, expr.span));
                    }
                    Ok(Expr {
                        ty: operands_ty,
                        span,
                        kind: ExprKind::Arithmetic(arithmetic_op, Box::new(lowered_lhs), Box::new(lowered_rhs)),
                    })
                }
                BinaryOp::LogicalOr => {
                    if let Some(expect_type) = expect_type
                        && !expect_type.is_bool()
                    {
                        return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().bool, expr.span));
                    }
                    let lowered_lhs = self.lower_expr(lhs, Some(self.ctx.types().bool))?;
                    let lowered_rhs = self.lower_expr(rhs, Some(self.ctx.types().bool))?;
                    Ok(Expr {
                        ty: self.ctx.types().bool,
                        span,
                        kind: ExprKind::If {
                            cond: Box::new(lowered_lhs),
                            if_true: Box::new(Expr::const_bool(self.ctx, true)),
                            if_false: Box::new(lowered_rhs),
                        },
                    })
                }
                BinaryOp::LogicalAnd => {
                    if let Some(expect_type) = expect_type
                        && !expect_type.is_bool()
                    {
                        return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().bool, expr.span));
                    }
                    let lowered_lhs = self.lower_expr(lhs, Some(self.ctx.types().bool))?;
                    let lowered_rhs = self.lower_expr(rhs, Some(self.ctx.types().bool))?;
                    Ok(Expr {
                        ty: self.ctx.types().bool,
                        span,
                        kind: ExprKind::If {
                            cond: Box::new(lowered_lhs),
                            if_true: Box::new(lowered_rhs),
                            if_false: Box::new(Expr::const_bool(self.ctx, false)),
                        },
                    })
                }
            },
            ast::ExprKind::Unary(op, rhs) => match *op {
                ast::UnaryOp::Negate => {
                    let lowered_rhs = self.lower_expr(rhs, None)?; // todo: pass expected type
                    let int_ty = match lowered_rhs.ty.as_int() {
                        Some(i) if i.is_signed() => i,
                        _ => {
                            return Err(Error::new(format!(
                                "only signed integer can be negated, not {:?}",
                                lowered_rhs.ty
                            ))
                            .with_span(expr.span));
                        }
                    };
                    if let Some(expect_type) = expect_type
                        && expect_type != lowered_rhs.ty
                    {
                        return Err(Error::expr_type_mismatch(expect_type, lowered_rhs.ty, expr.span));
                    }
                    Ok(Expr {
                        ty: lowered_rhs.ty,
                        span,
                        kind: ExprKind::Arithmetic(
                            ArithmeticOp::Sub,
                            Box::new(Expr::new_const(self.ctx, Constant::int(self.ctx, 0, int_ty).unwrap())),
                            Box::new(lowered_rhs),
                        ),
                    })
                }
                ast::UnaryOp::Not => {
                    if let Some(expect_type) = expect_type
                        && !expect_type.is_bool()
                    {
                        return Err(Error::expr_type_mismatch(expect_type, self.ctx.types().bool, expr.span));
                    }
                    let lowered_rhs = self.lower_expr(rhs, Some(self.ctx.types().bool))?;
                    Ok(Expr {
                        ty: self.ctx.types().bool,
                        span,
                        kind: ExprKind::Not(Box::new(lowered_rhs)),
                    })
                }
                ast::UnaryOp::AddressOf => {
                    let lowered_rhs = self.lower_expr(rhs, None)?.expect_place()?;
                    let ty = Type::new(
                        self.ctx,
                        TypeInfo::Ptr {
                            pointee: Some(lowered_rhs.ty),
                        },
                    );
                    if let Some(expect_type) = expect_type
                        && expect_type != ty
                    {
                        return Err(Error::expr_type_mismatch(expect_type, ty, expr.span));
                    }
                    Ok(Expr {
                        ty,
                        span,
                        kind: ExprKind::GetPointer(lowered_rhs),
                    })
                }
                ast::UnaryOp::Dereference => {
                    let expect_ptr_ty = expect_type.map(|ty| Type::new(self.ctx, TypeInfo::Ptr { pointee: Some(ty) }));
                    let lowered_ptr = self.lower_expr(rhs, expect_ptr_ty)?;
                    let ty = match lowered_ptr.ty.info() {
                        TypeInfo::Ptr { pointee } => pointee
                            .ok_or_else(|| Error::new("cannot dereference an opaque pointer").with_span(expr.span))?,
                        other => {
                            return Err(
                                Error::new(format!("expected an expression of type pointer, got {other:?}"))
                                    .with_span(rhs.span),
                            );
                        }
                    };
                    Ok(Expr {
                        ty,
                        span,
                        kind: ExprKind::Load(Place {
                            ty,
                            span,
                            kind: PlaceKind::Dereference(Box::new(lowered_ptr)),
                        }),
                    })
                }
            },
            ast::ExprKind::AsCast(value, ty) => {
                let ty = type_from_ast(self.ctx, self.types_scope, ty)?;
                if let Some(expect_type) = expect_type
                    && expect_type != ty
                {
                    return Err(Error::expr_type_mismatch(expect_type, ty, expr.span));
                }
                let lowered_expr = self.lower_expr(value, None)?;
                if (lowered_expr.ty.is_int() && ty.is_int()) || (lowered_expr.ty.is_ptr() && ty.is_ptr()) {
                    Ok(Expr {
                        ty,
                        span,
                        kind: ExprKind::Cast(Box::new(lowered_expr)),
                    })
                } else {
                    Err(Error::new(format!("cannot cast from {:?} to {:?}", lowered_expr.ty, ty)).with_span(expr.span))
                }
            }
            ast::ExprKind::TypeAscription(value, ty) => {
                let ty = type_from_ast(self.ctx, self.types_scope, ty)?;
                if let Some(expect_type) = expect_type
                    && expect_type != ty
                {
                    return Err(Error::expr_type_mismatch(expect_type, ty, expr.span));
                }
                let lowered_value = self.lower_expr(value, Some(ty))?;
                Ok(Expr { span, ..lowered_value })
            }
            ast::ExprKind::Comptime(cexpr) => {
                let lowered_cexpr = self.lower_expr(cexpr, expect_type)?;
                Ok(Expr {
                    ty: lowered_cexpr.ty,
                    span,
                    kind: ExprKind::Comptime(Box::new(lowered_cexpr)),
                })
            }
            ast::ExprKind::Range(_from, _to) => Err(Error::new(
                "range expressions are only allowed in for loops: 'for i in a..b {}'",
            )
            .with_span(expr.span)),
            ast::ExprKind::FieldAccess(lhs, field) => {
                let lowered_lhs = self.lower_expr(lhs, None)?;
                let (struct_ty, struct_, type_arguments, need_deref) = match lowered_lhs.ty.info() {
                    TypeInfo::Struct {
                        struct_,
                        type_arguments,
                    } => (lowered_lhs.ty, *struct_, *type_arguments, false),
                    TypeInfo::Ptr { pointee: Some(pointee) } if pointee.as_struct().is_some() => {
                        let (struct_, type_arguments) = pointee.as_struct().unwrap();
                        (*pointee, struct_, type_arguments, true)
                    }
                    _ => return Err(Error::new("only structs have fields").with_span(expr.span)),
                };
                let (_, field_id) = struct_
                    .info()
                    .fields
                    .iter()
                    .find(|(name, _)| name.value == field.value)
                    .ok_or_else(|| {
                        Error::new(format!("struct {struct_:?} has no field {:?}", field.value)).with_span(field.span)
                    })?;
                let field_ty = self.ctx.type_of_struct_field(*field_id, type_arguments);
                if let Some(expect_type) = expect_type
                    && expect_type != field_ty
                {
                    return Err(Error::expr_type_mismatch(expect_type, field_ty, expr.span));
                }
                Ok(if need_deref {
                    let struct_place = Place {
                        ty: struct_ty,
                        span,
                        kind: PlaceKind::Dereference(Box::new(lowered_lhs)),
                    };
                    Expr {
                        ty: field_ty,
                        span,
                        kind: ExprKind::Load(Place {
                            ty: field_ty,
                            span,
                            kind: PlaceKind::Field(Box::new(struct_place), *field_id),
                        }),
                    }
                } else {
                    Expr {
                        ty: field_ty,
                        span,
                        kind: ExprKind::Field(Box::new(lowered_lhs), *field_id),
                    }
                })
            }
            ast::ExprKind::Index(lhs, index) => {
                let lowered_lhs = self.lower_expr(lhs, None)?;
                let (element_ty, _length) = lowered_lhs.ty.as_array().ok_or_else(|| {
                    Error::new(format!("expected an array, got {:?}", lowered_lhs.ty)).with_span(lhs.span)
                })?;
                if let Some(expect_type) = expect_type
                    && expect_type != element_ty
                {
                    return Err(Error::expr_type_mismatch(expect_type, element_ty, expr.span));
                }
                let lowered_index = self.lower_expr(index, Some(self.ctx.types().usize))?;
                Ok(Expr {
                    ty: element_ty,
                    span,
                    kind: ExprKind::ArrayElement(Box::new(lowered_lhs), Box::new(lowered_index)),
                })
            }
        }
    }

    /// Lower a block expression
    fn lower_block_expr(
        &mut self,
        expr: &ast::BlockExpr,
        expect_type: Option<Type<'ctx>>,
    ) -> Result<Expr<'ctx>, Error> {
        if expr.final_expr.is_none()
            && let Some(expect_type) = expect_type
            && !expect_type.is_unit()
        {
            return Err(
                Error::new(format!("expected expr of type {expect_type:?}, found end-of-block"))
                    .with_span(expr.closing_brace_span),
            );
        }

        self.scope.push();

        let mut variables = Vec::new();
        let mut exprs = Vec::new();

        for stmt in &expr.statements {
            match stmt {
                ast::Statement::Let(let_statement) => match let_statement {
                    ast::LetStatement::WithValue { name, ty, value } => {
                        let var_id = VariableId::new();
                        let var_ty = ty
                            .as_ref()
                            .map(|ty| type_from_ast(self.ctx, self.types_scope, ty))
                            .transpose()?;
                        let value_eval = self.lower_expr(value, var_ty)?;
                        variables.push(VariableDeclaration {
                            id: var_id,
                            ty: value_eval.ty,
                            debug_name: name.value.clone(),
                        });
                        self.scope.variables.insert(name.value.clone(), (var_id, value_eval.ty));
                        exprs.push(Expr::set_var(self.ctx, var_id, value_eval));
                    }
                    ast::LetStatement::WithoutValue { name, ty } => {
                        let id = VariableId::new();
                        let ty = type_from_ast(self.ctx, self.types_scope, ty)?;
                        variables.push(VariableDeclaration {
                            id,
                            ty,
                            debug_name: name.value.clone(),
                        });
                        self.scope.variables.insert(name.value.clone(), (id, ty));
                    }
                },
                ast::Statement::Expr(expr) => {
                    exprs.push(self.lower_expr(expr, None)?);
                }
            }
        }

        match &expr.final_expr {
            Some(final_expr) => {
                exprs.push(self.lower_expr(final_expr, expect_type)?);
            }
            None => {
                if !exprs.is_empty() {
                    exprs.push(Expr::unit(self.ctx));
                }
            }
        }

        self.scope.pop();

        Ok(Expr {
            ty: exprs.last().map_or_else(|| self.ctx.types().unit, |expr| expr.ty),
            span: Some(expr.span()),
            kind: ExprKind::Block(BlockExpr { variables, exprs }),
        })
    }

    fn lower_loop_body(
        &mut self,
        body: &ast::BlockExpr,
        expect_type: Option<Type<'ctx>>,
    ) -> Result<LowerLoopBodyResult<'ctx>, Error> {
        let loop_id = LoopId::new();
        self.scope.push();
        self.scope.loop_context = Some(LoopContext {
            loop_id,
            break_used_with_type: None,
            expect_type,
        });
        let body = self.lower_block_expr(body, Some(self.ctx.types().unit))?;
        let loop_ctx = self.scope.loop_context.unwrap();
        self.scope.pop();
        Ok(LowerLoopBodyResult {
            loop_id,
            body,
            break_used_with_type: loop_ctx.break_used_with_type,
        })
    }
}

fn coalesce_types<'ctx>(a: Type<'ctx>, b: Type<'ctx>) -> Type<'ctx> {
    if a.is_never() { b } else { a }
}
