use super::*;
use crate::ir_tree::visit::ExprVisitor;

pub fn purity_check(body: &Expr, functions: &BTreeMap<FunctionId, Function>) -> Result<(), Error> {
    struct Visitor<'a> {
        error: Option<Error>,
        functions: &'a BTreeMap<FunctionId, Function>,
    }

    impl ExprVisitor for Visitor<'_> {
        fn visit_expr(&mut self, expr: &Expr) {
            if self.error.is_some() {
                return;
            }

            match &expr.kind {
                ExprKind::Undefined
                | ExprKind::ConstUnit
                | ExprKind::ConstNumber(_)
                | ExprKind::ConstString(_)
                | ExprKind::ConstBool(_)
                | ExprKind::Load(_)
                | ExprKind::Field(_, _)
                | ExprKind::ArrayElement(_, _)
                | ExprKind::Store(_, _)
                | ExprKind::Argument(_)
                | ExprKind::Block(_)
                | ExprKind::Return(_)
                | ExprKind::Break(_, _)
                | ExprKind::Arithmetic(_, _, _)
                | ExprKind::Cmp(_, _, _)
                | ExprKind::If { .. }
                | ExprKind::Loop(_, _)
                | ExprKind::ArrayInitializer(_)
                | ExprKind::StructInitializer(_)
                | ExprKind::Cast(_)
                | ExprKind::Not(_) => (),
                ExprKind::GetPointer(_) => {
                    self.error =
                        Some(Error::new("getting pointers is not a pure operation").with_span(expr.span.unwrap()));
                    return;
                }
                ExprKind::FunctionCall(function_id, _) => match self.functions[function_id].is_pure {
                    true => (),
                    false => {
                        self.error = Some(
                            Error::new("only pure functions may be called from pure functions")
                                .with_span(expr.span.unwrap()),
                        );
                        return;
                    }
                },
            }

            expr.visit_children(self);
        }

        fn visit_place(&mut self, place: &Place) {
            match &place.kind {
                PlaceKind::Variable(_) | PlaceKind::Field(_, _) | PlaceKind::ArrayElement(_, _) => (),
                PlaceKind::Dereference(_) => {
                    self.error =
                        Some(Error::new("dereferencing is not a pure operation").with_span(place.span.unwrap()));
                    return;
                }
            }

            place.visit_children(self);
        }
    }

    let mut visitor = Visitor { error: None, functions };
    visitor.visit_expr(body);

    match visitor.error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}
