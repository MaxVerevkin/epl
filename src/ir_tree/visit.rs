use super::*;

impl ExprRef<'_> {
    pub fn visit_children(&self, mut visitor: impl FnMut(Self)) {
        match self {
            Self::R(e) => match &e.kind {
                RExprKind::Undefined
                | RExprKind::ConstUnit
                | RExprKind::ConstNumber(_)
                | RExprKind::ConstString(_)
                | RExprKind::ConstBool(_)
                | RExprKind::Argument(_) => (),
                RExprKind::Field(rexpr, _) => visitor(Self::R(rexpr)),
                RExprKind::ArrayElement(rexpr, expr) => {
                    visitor(Self::R(rexpr));
                    visitor(expr.as_ref().as_ref());
                }
                RExprKind::Store(lexpr, expr) => {
                    visitor(Self::L(lexpr));
                    visitor(expr.as_ref().as_ref());
                }
                RExprKind::GetPointer(lexpr) => visitor(Self::L(lexpr)),
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
                RExprKind::BinOp(_, expr, expr1) => {
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
            Self::L(e) => match &e.kind {
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
            Self::R(e) => match &mut e.kind {
                RExprKind::Undefined
                | RExprKind::ConstUnit
                | RExprKind::ConstNumber(_)
                | RExprKind::ConstString(_)
                | RExprKind::ConstBool(_)
                | RExprKind::Argument(_) => (),
                RExprKind::Field(rexpr, _) => visitor(ExprMutRef::R(rexpr.as_mut())),
                RExprKind::ArrayElement(rexpr, expr) => {
                    visitor(ExprMutRef::R(rexpr.as_mut()));
                    visitor(expr.as_mut().as_mut());
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
                RExprKind::BinOp(_, expr, expr1) => {
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
            Self::L(e) => match &mut e.kind {
                LExprKind::Dereference(expr) => visitor(expr.as_mut().as_mut()),
                LExprKind::Variable(_) => (),
                LExprKind::Field(lexpr, _) => visitor(ExprMutRef::L(lexpr.as_mut())),
                LExprKind::ArrayElement(lexpr, expr) => {
                    visitor(ExprMutRef::L(lexpr.as_mut()));
                    visitor(expr.as_mut().as_mut());
                }
            },
        }
    }
}
