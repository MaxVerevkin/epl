use std::collections::HashSet;

use super::*;
use crate::ir_tree::visit::ExprVisitor;

/// Verify that `comptime` exprs are valid
pub fn check_comptime_exprs(body: &Expr, module: &Module) -> Result<(), Error> {
    struct Visitor<'a> {
        result: Result<(), Error>,
        module: &'a Module,
    }

    impl ExprVisitor<'_> for Visitor<'_> {
        fn visit_expr(&mut self, expr: &Expr) {
            if self.result.is_ok() {
                if let ExprKind::Comptime(comptime_expr) = &expr.kind {
                    self.result = purity_check(Context::ComptimeExpr, comptime_expr, self.module);
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
pub fn check_pure_function(body: &Expr, module: &Module) -> Result<(), Error> {
    purity_check(Context::PureFunctionBody, body, module)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Context {
    PureFunctionBody,
    ComptimeExpr,
}

fn purity_check(context: Context, expr: &Expr, module: &Module) -> Result<(), Error> {
    struct Visitor<'a> {
        result: Result<(), Error>,
        module: &'a Module,
        context: Context,
        in_scope_variables: HashSet<VariableId>,
        in_scope_loops: HashSet<LoopId>,
    }

    impl ExprVisitor<'_> for Visitor<'_> {
        fn visit_expr(&mut self, expr: &Expr) {
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
                    Context::PureFunctionBody => (),
                    Context::ComptimeExpr => {
                        self.result =
                            Err(Error::new("cannot access arguments from a comptime block")
                                .with_span(expr.span.unwrap()));
                    }
                },
                ExprKind::GetPointer(_) => {
                    self.result =
                        Err(Error::new("getting pointers is not a pure operation").with_span(expr.span.unwrap()));
                }
                ExprKind::FunctionCall(function_id, _) => match self.module.functions[function_id].is_pure {
                    true => (),
                    false => {
                        self.result = Err(Error::new("only pure functions may be called from pure functions")
                            .with_span(expr.span.unwrap()));
                    }
                },
                ExprKind::Comptime(comptime_expr) => match self.context {
                    Context::PureFunctionBody => (), // checked by `check_comptime_exprs`
                    Context::ComptimeExpr => {
                        // Comptimes are self-sufficient and do not depend on context
                        self.result = purity_check(Context::ComptimeExpr, comptime_expr, self.module);
                    }
                },
                ExprKind::Cast(from_expr) => {
                    if (from_expr.ty.is_int() && expr.ty.is_ptr()) || (from_expr.ty.is_ptr() && expr.ty.is_int()) {
                        self.result = Err(Error::new("casting pointers to integers and vise versa is not pure")
                            .with_span(expr.span.unwrap()));
                    }
                }
                ExprKind::Return(_) => match self.context {
                    Context::PureFunctionBody => (),
                    Context::ComptimeExpr => {
                        self.result =
                            Err(Error::new("cannot return from a comptime expr").with_span(expr.span.unwrap()));
                    }
                },
                ExprKind::Break(loop_id, _) | ExprKind::Continue(loop_id) => match self.context {
                    Context::PureFunctionBody => (),
                    Context::ComptimeExpr => {
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

        fn visit_place(&mut self, place: &Place) {
            if self.result.is_err() {
                return;
            }

            match &place.kind {
                PlaceKind::Variable(var_id) => match self.context {
                    Context::PureFunctionBody => (),
                    Context::ComptimeExpr => {
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
