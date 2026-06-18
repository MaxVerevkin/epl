use std::collections::HashSet;

use super::*;
use crate::ir_tree::visit::ExprVisitor;

pub fn run_checkers<'ctx>(function_id: FunctionId, module: &Module<'ctx>) -> Result<(), Error> {
    let function = &module.functions[&function_id];
    check_main_abi(module.ctx, function)?;
    check_intrinsic(function)?;
    check_comptime_exprs(function, module)?;
    check_pure_function(function, module)?;
    Ok(())
}

/// Verify that the signature of `main` is `fn main() -> i32`
fn check_main_abi<'ctx>(ctx: Context<'ctx>, function: &Function<'ctx>) -> Result<(), Error> {
    if function.name_ident.value != "main" {
        return Ok(());
    }

    if function.is_variadic || !function.args.is_empty() || function.return_ty != ctx.types().i32 {
        return Err(
            Error::new("incorrect 'main' function signature: must be 'fn main() -> i32'")
                .with_span(function.name_ident.span),
        );
    }

    Ok(())
}

/// Verify all known intrinsics
fn check_intrinsic<'ctx>(function: &Function<'ctx>) -> Result<(), Error> {
    if !function.is_intrinsic {
        return Ok(());
    }

    if function.body.is_some() {
        return Err(Error::new("intrinsics must not have bodies").with_span(function.name_ident.span));
    }

    match function.name_ident.value.as_str() {
        "size_of" => (),
        other => return Err(Error::new(format!("unknown intrinsic: {other:?}")).with_span(function.name_ident.span)),
    }

    Ok(())
}

/// Verify that `comptime` exprs are valid
fn check_comptime_exprs<'ctx>(function: &Function<'ctx>, module: &Module<'ctx>) -> Result<(), Error> {
    let Some(body) = &function.body else { return Ok(()) };

    struct Visitor<'a, 'ctx> {
        result: Result<(), Error>,
        module: &'a Module<'ctx>,
    }

    impl<'a, 'ctx> ExprVisitor<'a, 'ctx> for Visitor<'a, 'ctx> {
        fn visit_expr(&mut self, expr: &'a Expr<'ctx>) {
            if self.result.is_ok() {
                if let ExprKind::Comptime(comptime_expr) = &expr.kind {
                    self.result = purity_check(PurityContext::ComptimeExpr, comptime_expr, self.module);
                } else {
                    expr.visit_children(self);
                }
            }
        }
    }

    let mut visitor = Visitor { result: Ok(()), module };
    visitor.visit_expr(body);
    visitor.result
}

/// Verify that `@pure` functions are valid
fn check_pure_function<'ctx>(function: &Function<'ctx>, module: &Module<'ctx>) -> Result<(), Error> {
    if !function.is_pure {
        return Ok(());
    }
    match &function.body {
        Some(body) => purity_check(PurityContext::PureFunctionBody, body, module),
        None if function.is_intrinsic => Ok(()),
        None => Err(Error::new("pure functions must have a body").with_span(function.name_ident.span)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PurityContext {
    PureFunctionBody,
    ComptimeExpr,
}

fn purity_check<'ctx>(context: PurityContext, expr: &Expr<'ctx>, module: &Module<'ctx>) -> Result<(), Error> {
    struct Visitor<'a, 'ctx> {
        result: Result<(), Error>,
        module: &'a Module<'ctx>,
        context: PurityContext,
        in_scope_variables: HashSet<VariableId>,
        in_scope_loops: HashSet<LoopId>,
    }

    impl<'a, 'ctx> ExprVisitor<'a, 'ctx> for Visitor<'a, 'ctx> {
        fn visit_expr(&mut self, expr: &'a Expr<'ctx>) {
            if self.result.is_err() {
                return;
            }

            match &expr.kind {
                ExprKind::Const(..)
                | ExprKind::Field(_, _)
                | ExprKind::ArrayElement(_, _)
                | ExprKind::Load(_)
                | ExprKind::Store(_, _)
                | ExprKind::Arithmetic(_, _, _)
                | ExprKind::InPlaceArithmetic(_, _, _)
                | ExprKind::Cmp(_, _, _)
                | ExprKind::If { .. }
                | ExprKind::ArrayInitializer(_)
                | ExprKind::StructInitializer(_)
                | ExprKind::Not(_) => (),
                ExprKind::ConstString(_) => {
                    self.result = Err(Error::new("constant strings in pure functions are not yet supported")
                        .with_span(expr.span.unwrap()));
                }
                ExprKind::Argument(_) => match self.context {
                    PurityContext::PureFunctionBody => (),
                    PurityContext::ComptimeExpr => {
                        self.result =
                            Err(Error::new("cannot access arguments from a comptime block")
                                .with_span(expr.span.unwrap()));
                    }
                },
                ExprKind::GetPointer(_) => {
                    self.result =
                        Err(Error::new("getting pointers is not a pure operation").with_span(expr.span.unwrap()));
                }
                ExprKind::FunctionCall(function_id, _, _) => match self.module.functions[function_id].is_pure {
                    true => (),
                    false => {
                        self.result = Err(Error::new("only pure functions may be called from pure functions")
                            .with_span(expr.span.unwrap()));
                    }
                },
                ExprKind::Comptime(comptime_expr) => match self.context {
                    PurityContext::PureFunctionBody => (), // checked by `check_comptime_exprs`
                    PurityContext::ComptimeExpr => {
                        // Comptimes are self-sufficient and do not depend on context
                        self.result = purity_check(PurityContext::ComptimeExpr, comptime_expr, self.module);
                    }
                },
                ExprKind::Cast(from_expr) => {
                    if (from_expr.ty.is_int() && expr.ty.is_ptr()) || (from_expr.ty.is_ptr() && expr.ty.is_int()) {
                        self.result = Err(Error::new("casting pointers to integers and vice versa is not pure")
                            .with_span(expr.span.unwrap()));
                    }
                }
                ExprKind::Return(_) => match self.context {
                    PurityContext::PureFunctionBody => (),
                    PurityContext::ComptimeExpr => {
                        self.result =
                            Err(Error::new("cannot return from a comptime expr").with_span(expr.span.unwrap()));
                    }
                },
                ExprKind::Break(loop_id, _) | ExprKind::Continue(loop_id) => match self.context {
                    PurityContext::PureFunctionBody => (),
                    PurityContext::ComptimeExpr => {
                        if !self.in_scope_loops.contains(loop_id) {
                            self.result =
                                Err(Error::new("loop out of context of this comptime expr")
                                    .with_span(expr.span.unwrap()));
                        }
                    }
                },
                ExprKind::Block(block) => {
                    for var in &block.variables {
                        self.in_scope_variables.insert(var.id);
                    }
                    expr.visit_children(self);
                    for var in &block.variables {
                        self.in_scope_variables.remove(&var.id);
                    }
                    return;
                }
                ExprKind::Loop(loop_id, body) => {
                    self.in_scope_loops.insert(*loop_id);
                    self.visit_expr(body);
                    self.in_scope_loops.remove(loop_id);
                    return;
                }
            }

            if self.result.is_ok() {
                expr.visit_children(self);
            }
        }

        fn visit_place(&mut self, place: &'a Place<'ctx>) {
            if self.result.is_err() {
                return;
            }

            match &place.kind {
                PlaceKind::Variable(var_id) => match self.context {
                    PurityContext::PureFunctionBody => (),
                    PurityContext::ComptimeExpr => {
                        if !self.in_scope_variables.contains(var_id) {
                            self.result = Err(Error::new("variable out of context of this comptime expr")
                                .with_span(place.span.unwrap()));
                        }
                    }
                },
                PlaceKind::Field(_, _) | PlaceKind::ArrayElement(_, _) => (),
                PlaceKind::Dereference(_) => {
                    self.result =
                        Err(Error::new("dereferencing is not a pure operation").with_span(place.span.unwrap()));
                }
            }

            if self.result.is_ok() {
                place.visit_children(self);
            }
        }
    }

    let mut visitor = Visitor {
        result: Ok(()),
        module,
        context,
        in_scope_variables: HashSet::new(),
        in_scope_loops: HashSet::new(),
    };
    visitor.visit_expr(expr);
    visitor.result
}
