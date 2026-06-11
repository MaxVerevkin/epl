use super::*;

pub trait ExprVisitor<'a, 'ctx>: Sized {
    fn visit_expr(&mut self, expr: &'a Expr<'ctx>) {
        expr.visit_children(self);
    }

    fn visit_place(&mut self, place: &'a Place<'ctx>) {
        place.visit_children(self);
    }
}

pub trait ExprVisitorMut<'a, 'ctx>: Sized {
    fn visit_expr(&mut self, expr: &'a mut Expr<'ctx>) {
        expr.visit_children_mut(self);
    }

    fn visit_place(&mut self, place: &'a mut Place<'ctx>) {
        place.visit_children_mut(self);
    }
}

impl<'ctx> Expr<'ctx> {
    pub fn visit_children<'a>(&'a self, visitor: &mut impl ExprVisitor<'a, 'ctx>) {
        match &self.kind {
            ExprKind::Const(_) | ExprKind::ConstString(_) | ExprKind::Argument(_) | ExprKind::Continue(_) => (),

            ExprKind::Field(expr, _)
            | ExprKind::Return(expr)
            | ExprKind::Break(_, expr)
            | ExprKind::Loop(_, expr)
            | ExprKind::Cast(expr)
            | ExprKind::Not(expr)
            | ExprKind::Comptime(expr) => visitor.visit_expr(expr.as_ref()),

            ExprKind::ArrayElement(expr1, expr2)
            | ExprKind::Arithmetic(_, expr1, expr2)
            | ExprKind::Cmp(_, expr1, expr2) => {
                visitor.visit_expr(expr1.as_ref());
                visitor.visit_expr(expr2.as_ref());
            }

            ExprKind::Store(place, value) | ExprKind::InPlaceArithmetic(_, place, value) => {
                visitor.visit_place(place);
                visitor.visit_expr(value.as_ref());
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
                visitor.visit_expr(cond.as_ref());
                visitor.visit_expr(if_true.as_ref());
                visitor.visit_expr(if_false.as_ref());
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

    pub fn visit_children_mut<'a>(&'a mut self, visitor: &mut impl ExprVisitorMut<'a, 'ctx>) {
        match &mut self.kind {
            ExprKind::Const(_) | ExprKind::ConstString(_) | ExprKind::Argument(_) | ExprKind::Continue(_) => (),

            ExprKind::Field(expr, _)
            | ExprKind::Return(expr)
            | ExprKind::Break(_, expr)
            | ExprKind::Loop(_, expr)
            | ExprKind::Cast(expr)
            | ExprKind::Not(expr)
            | ExprKind::Comptime(expr) => visitor.visit_expr(&mut *expr),

            ExprKind::ArrayElement(expr1, expr2)
            | ExprKind::Arithmetic(_, expr1, expr2)
            | ExprKind::Cmp(_, expr1, expr2) => {
                visitor.visit_expr(&mut *expr1);
                visitor.visit_expr(&mut *expr2);
            }

            ExprKind::Store(place, value) | ExprKind::InPlaceArithmetic(_, place, value) => {
                visitor.visit_place(place);
                visitor.visit_expr(&mut *value);
            }

            ExprKind::Load(place) | ExprKind::GetPointer(place) => visitor.visit_place(place),

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

impl<'ctx> Place<'ctx> {
    pub fn visit_children<'a>(&'a self, visitor: &mut impl ExprVisitor<'a, 'ctx>) {
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

    pub fn visit_children_mut<'a>(&'a mut self, visitor: &mut impl ExprVisitorMut<'a, 'ctx>) {
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
