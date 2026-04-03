use super::*;

impl ExprMutRef<'_> {
    /// Collapses redundantly nested expressions, eliminates dead code.
    pub fn basic_optimize(&mut self) {
        self.visit_children(|mut expr| expr.basic_optimize());

        if let Some(this_lexpr) = self.as_lexpr()
            && let LExprKind::Dereference(ptr_expr) = &mut this_lexpr.kind
            && let Expr::R(ptr_rexpr) = ptr_expr.as_mut()
            && let RExprKind::GetPointer(place_expr) = &mut ptr_rexpr.kind
        {
            let place_expr = std::mem::replace(place_expr.as_mut(), LExpr::DUMMY);
            *this_lexpr = place_expr;
        }

        if let ExprMutRef::Any(this_expr) = self
            && let Expr::R(this_rexpr) = this_expr
            && let RExprKind::GetPointer(place_lexpr) = &mut this_rexpr.kind
            && let LExprKind::Dereference(ptr_expr) = &mut place_lexpr.kind
        {
            let ptr_expr = std::mem::replace(ptr_expr.as_mut(), Expr::DUMMY);
            **this_expr = ptr_expr;
        }

        if let ExprMutRef::Any(this_expr) = self
            && let Expr::R(this_rexpr) = this_expr
            && let RExprKind::Block(block) = &mut this_rexpr.kind
        {
            block.eliminate_dead_code();
            if block.variables.is_empty() {
                if block.exprs.is_empty() {
                    this_rexpr.kind = RExprKind::ConstUnit;
                } else if block.exprs.len() == 1 && matches!(block.exprs[0], Expr::R(_)) {
                    // TODO: convert lexprs into rexprs
                    let Some(Expr::R(expr)) = block.exprs.pop() else {
                        unreachable!()
                    };
                    *this_rexpr = expr;
                }
            }
        }
    }
}

impl Expr {
    fn is_pure(&self) -> bool {
        if let Self::R(expr) = self
            && matches!(
                expr.kind,
                RExprKind::Undefined
                    | RExprKind::ConstUnit
                    | RExprKind::ConstNumber(_)
                    | RExprKind::ConstString(_)
                    | RExprKind::ConstBool(_)
            )
        {
            true
        } else {
            false
        }
    }

    fn is_const_unit(&self) -> bool {
        if let Self::R(expr) = self
            && matches!(expr.kind, RExprKind::ConstUnit)
        {
            true
        } else {
            false
        }
    }
}

impl BlockExpr {
    fn eliminate_dead_code(&mut self) {
        // remove all pure exprs except the last one
        let mut expr_i = 0;
        let exprs_len = self.exprs.len();
        self.exprs.retain(|expr| {
            let remove = expr.is_pure() && expr_i + 1 != exprs_len;
            expr_i += 1;
            !remove
        });

        // remove trailing unit if expr before that is of unit type
        if !self.exprs.is_empty()
            && self.exprs[self.exprs.len() - 1].is_const_unit()
            && (self.exprs.len() == 1 || self.exprs[self.exprs.len() - 2].ty() == Type::Unit)
        {
            self.exprs.pop();
        }
    }
}
