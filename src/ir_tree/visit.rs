use super::*;

pub trait ExprVisitor: Sized {
    fn visit_expr(&mut self, expr: &Expr) {
        expr.visit_children(self);
    }

    fn visit_place(&mut self, place: &Place) {
        place.visit_children(self);
    }
}

pub trait ExprVisitorMut: Sized {
    fn visit_expr(&mut self, expr: &mut Expr) {
        expr.visit_children_mut(self);
    }

    fn visit_place(&mut self, place: &mut Place) {
        place.visit_children_mut(self);
    }
}

impl Expr {
    pub fn visit_children(&self, visitor: &mut impl ExprVisitor) {
        match &self.kind {
            ExprKind::Undefined
            | ExprKind::ConstUnit
            | ExprKind::ConstNumber(_)
            | ExprKind::ConstString(_)
            | ExprKind::ConstBool(_)
            | ExprKind::Argument(_) => (),

            ExprKind::Field(expr, _)
            | ExprKind::Return(expr)
            | ExprKind::Break(_, expr)
            | ExprKind::Loop(_, expr)
            | ExprKind::Cast(expr)
            | ExprKind::Not(expr) => visitor.visit_expr(expr),

            ExprKind::ArrayElement(expr1, expr2)
            | ExprKind::Arithmetic(_, expr1, expr2)
            | ExprKind::Cmp(_, expr1, expr2) => {
                visitor.visit_expr(expr1);
                visitor.visit_expr(expr2);
            }

            ExprKind::Store(place, value) => {
                visitor.visit_place(place);
                visitor.visit_expr(value);
            }

            ExprKind::Load(place) | ExprKind::GetPointer(place) => visitor.visit_place(place),

            ExprKind::Block(bexpr) => {
                for expr in &bexpr.exprs {
                    visitor.visit_expr(expr);
                }
            }

            ExprKind::If {
                cond,
                if_true,
                if_false,
            } => {
                visitor.visit_expr(cond);
                visitor.visit_expr(if_true);
                visitor.visit_expr(if_false);
            }

            ExprKind::ArrayInitializer(exprs) | ExprKind::FunctionCall(_, exprs) => {
                for expr in exprs {
                    visitor.visit_expr(expr);
                }
            }

            ExprKind::StructInitializer(items) => {
                for (_, expr) in items {
                    visitor.visit_expr(expr);
                }
            }
        }
    }

    pub fn visit_children_mut(&mut self, visitor: &mut impl ExprVisitorMut) {
        match &mut self.kind {
            ExprKind::Undefined
            | ExprKind::ConstUnit
            | ExprKind::ConstNumber(_)
            | ExprKind::ConstString(_)
            | ExprKind::ConstBool(_)
            | ExprKind::Argument(_) => (),

            ExprKind::Field(expr, _)
            | ExprKind::Return(expr)
            | ExprKind::Break(_, expr)
            | ExprKind::Loop(_, expr)
            | ExprKind::Cast(expr)
            | ExprKind::Not(expr) => visitor.visit_expr(&mut *expr),

            ExprKind::ArrayElement(expr1, expr2)
            | ExprKind::Arithmetic(_, expr1, expr2)
            | ExprKind::Cmp(_, expr1, expr2) => {
                visitor.visit_expr(&mut *expr1);
                visitor.visit_expr(&mut *expr2);
            }

            ExprKind::Store(place, value) => {
                visitor.visit_place(&mut *place);
                visitor.visit_expr(&mut *value);
            }

            ExprKind::Load(place) | ExprKind::GetPointer(place) => visitor.visit_place(&mut *place),

            ExprKind::Block(bexpr) => {
                for expr in &mut bexpr.exprs {
                    visitor.visit_expr(&mut *expr);
                }
            }

            ExprKind::If {
                cond,
                if_true,
                if_false,
            } => {
                visitor.visit_expr(&mut *cond);
                visitor.visit_expr(&mut *if_true);
                visitor.visit_expr(&mut *if_false);
            }

            ExprKind::ArrayInitializer(exprs) | ExprKind::FunctionCall(_, exprs) => {
                for expr in exprs {
                    visitor.visit_expr(expr);
                }
            }

            ExprKind::StructInitializer(items) => {
                for (_, expr) in items {
                    visitor.visit_expr(expr);
                }
            }
        }
    }
}

impl Place {
    pub fn visit_children(&self, visitor: &mut impl ExprVisitor) {
        match &self.kind {
            PlaceKind::Variable(_) => (),
            PlaceKind::Dereference(ptr) => visitor.visit_expr(ptr),
            PlaceKind::Field(place, _) => visitor.visit_place(place),
            PlaceKind::ArrayElement(array, index) => {
                visitor.visit_place(array);
                visitor.visit_expr(index);
            }
        }
    }

    pub fn visit_children_mut(&mut self, visitor: &mut impl ExprVisitorMut) {
        match &mut self.kind {
            PlaceKind::Variable(_) => (),
            PlaceKind::Dereference(ptr) => visitor.visit_expr(&mut *ptr),
            PlaceKind::Field(place, _) => visitor.visit_place(&mut *place),
            PlaceKind::ArrayElement(array, index) => {
                visitor.visit_place(&mut *array);
                visitor.visit_expr(&mut *index);
            }
        }
    }
}
