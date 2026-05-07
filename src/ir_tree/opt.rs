use super::*;
use crate::ir_tree::visit::ExprVisitorMut;

/// Collapses redundantly nested expressions, eliminates dead code.
pub struct BasicOptVisitor;

impl ExprVisitorMut for BasicOptVisitor {
    fn visit_expr(&mut self, expr: &mut Expr) {
        expr.visit_children_mut(self);

        // GET_POINTER
        //   DEREFERENCE
        //     <expr>
        //
        // ->
        //
        // <expr>
        if let ExprKind::GetPointer(place) = &mut expr.kind
            && let PlaceKind::Dereference(ptr_expr) = &mut place.kind
        {
            let ptr_expr = std::mem::replace(ptr_expr.as_mut(), Expr::UNIT);
            *expr = ptr_expr;
        }

        // Simplify empty blocks and needlessly nested blocks
        if let ExprKind::Block(block) = &mut expr.kind {
            block.eliminate_dead_code();
            if block.variables.is_empty() {
                if block.exprs.is_empty() {
                    expr.kind = ExprKind::Const(Constant::Unit);
                } else if block.exprs.len() == 1 {
                    *expr = block.exprs.pop().unwrap();
                }
            }
        }
    }

    fn visit_place(&mut self, place: &mut Place) {
        place.visit_children_mut(self);

        // DEREFERENCE
        //   GET_POINTER
        //     <place>
        //
        // ->
        //
        // <place>
        if let PlaceKind::Dereference(ptr_expr) = &mut place.kind
            && let ExprKind::GetPointer(place_expr) = &mut ptr_expr.kind
        {
            let place_expr = std::mem::replace(place_expr, Place::DUMMY);
            *place = place_expr;
        }
    }
}

impl Expr {
    fn is_pure(&self) -> bool {
        matches!(
            self.kind,
            ExprKind::Const(_) | ExprKind::ConstString(_) | ExprKind::Argument(_)
        )
    }

    fn is_const_unit(&self) -> bool {
        matches!(self.kind, ExprKind::Const(Constant::Unit))
    }
}

impl BlockExpr {
    fn eliminate_dead_code(&mut self) {
        // remove all code after first expr with never type
        if let Some(first_never_expr_i) = self.exprs.iter().position(|expr| expr.ty == Type::Never) {
            self.exprs.truncate(first_never_expr_i + 1);
        }

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
            && (self.exprs.len() == 1 || self.exprs[self.exprs.len() - 2].ty == Type::Unit)
        {
            self.exprs.pop();
        }
    }
}
