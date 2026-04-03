use super::*;

impl ExprRef<'_> {
    pub fn visit_children(&self, mut visitor: impl FnMut(Self)) {
        match self {
            Self::Any(Expr::R(e)) => match &e.kind {
                RExprKind::Undefined
                | RExprKind::ConstUnit
                | RExprKind::ConstNumber(_)
                | RExprKind::ConstString(_)
                | RExprKind::ConstBool(_)
                | RExprKind::Argument(_) => (),
                RExprKind::Field(expr, _) => visitor(expr.as_ref().as_ref()),
                RExprKind::ArrayElement(array, index) => {
                    visitor(array.as_ref().as_ref());
                    visitor(index.as_ref().as_ref());
                }
                RExprKind::Store(lexpr, expr) => {
                    visitor(ExprRef::L(lexpr.as_ref()));
                    visitor(expr.as_ref().as_ref());
                }
                RExprKind::GetPointer(lexpr) => visitor(ExprRef::L(lexpr.as_ref())),
                RExprKind::Block(bexpr) => {
                    for expr in &bexpr.exprs {
                        visitor(expr.as_ref());
                    }
                }
                RExprKind::Return(expr)
                | RExprKind::Break(_, expr)
                | RExprKind::Loop(_, expr)
                | RExprKind::Cast(expr)
                | RExprKind::Not(expr) => visitor(expr.as_ref().as_ref()),
                RExprKind::Arithmetic(_, expr, expr1) | RExprKind::Cmp(_, expr, expr1) => {
                    visitor(expr.as_ref().as_ref());
                    visitor(expr1.as_ref().as_ref());
                }
                RExprKind::If {
                    cond,
                    if_true,
                    if_false,
                } => {
                    visitor(cond.as_ref().as_ref());
                    visitor(if_true.as_ref().as_ref());
                    visitor(if_false.as_ref().as_ref());
                }
                RExprKind::ArrayInitializer(exprs) | RExprKind::FunctionCall(_, exprs) => {
                    for expr in exprs {
                        visitor(expr.as_ref());
                    }
                }
                RExprKind::StructInitializer(items) => {
                    for (_, expr) in items {
                        visitor(expr.as_ref());
                    }
                }
            },
            Self::Any(Expr::L(e)) | &Self::L(e) => match &e.kind {
                LExprKind::Dereference(expr) => visitor(expr.as_ref().as_ref()),
                LExprKind::Variable(_) => (),
                LExprKind::Field(lexpr, _) => visitor(Self::L(lexpr)),
                LExprKind::ArrayElement(lexpr, expr) => {
                    visitor(Self::L(lexpr));
                    visitor(expr.as_ref().as_ref());
                }
            },
        }
    }
}

impl ExprMutRef<'_> {
    pub fn visit_children(&mut self, mut visitor: impl FnMut(ExprMutRef<'_>)) {
        match self {
            Self::Any(Expr::R(e)) => match &mut e.kind {
                RExprKind::Undefined
                | RExprKind::ConstUnit
                | RExprKind::ConstNumber(_)
                | RExprKind::ConstString(_)
                | RExprKind::ConstBool(_)
                | RExprKind::Argument(_) => (),
                RExprKind::Field(expr, _) => visitor(expr.as_mut().as_mut()),
                RExprKind::ArrayElement(array, index) => {
                    visitor(array.as_mut().as_mut());
                    visitor(index.as_mut().as_mut());
                }
                RExprKind::Store(lexpr, expr) => {
                    visitor(ExprMutRef::L(lexpr.as_mut()));
                    visitor(expr.as_mut().as_mut());
                }
                RExprKind::GetPointer(lexpr) => visitor(ExprMutRef::L(lexpr.as_mut())),
                RExprKind::Block(bexpr) => {
                    for expr in &mut bexpr.exprs {
                        visitor(expr.as_mut());
                    }
                }
                RExprKind::Return(expr)
                | RExprKind::Break(_, expr)
                | RExprKind::Loop(_, expr)
                | RExprKind::Cast(expr)
                | RExprKind::Not(expr) => visitor(expr.as_mut().as_mut()),
                RExprKind::Arithmetic(_, expr, expr1) | RExprKind::Cmp(_, expr, expr1) => {
                    visitor(expr.as_mut().as_mut());
                    visitor(expr1.as_mut().as_mut());
                }
                RExprKind::If {
                    cond,
                    if_true,
                    if_false,
                } => {
                    visitor(cond.as_mut().as_mut());
                    visitor(if_true.as_mut().as_mut());
                    visitor(if_false.as_mut().as_mut());
                }
                RExprKind::ArrayInitializer(exprs) | RExprKind::FunctionCall(_, exprs) => {
                    for expr in exprs {
                        visitor(expr.as_mut());
                    }
                }
                RExprKind::StructInitializer(items) => {
                    for (_, expr) in items {
                        visitor(expr.as_mut());
                    }
                }
            },
            Self::Any(Expr::L(e)) => visit_children_lexpr_mut(e, visitor),
            Self::L(e) => visit_children_lexpr_mut(e, visitor),
        }
    }
}

fn visit_children_lexpr_mut(lexpr: &mut LExpr, mut visitor: impl FnMut(ExprMutRef<'_>)) {
    match &mut lexpr.kind {
        LExprKind::Dereference(expr) => visitor(expr.as_mut().as_mut()),
        LExprKind::Variable(_) => (),
        LExprKind::Field(lexpr, _) => visitor(ExprMutRef::L(lexpr)),
        LExprKind::ArrayElement(lexpr, expr) => {
            visitor(ExprMutRef::L(lexpr));
            visitor(expr.as_mut().as_mut());
        }
    }
}
