use std::collections::BTreeSet;

use super::*;

pub fn collect_for_monomorphization<'ctx>(module: &Module<'ctx>) -> BTreeSet<(FunctionId, TypeArguments<'ctx>)> {
    struct CollectVisitor<'a, 'ctx> {
        ctx: Context<'ctx>,
        queue: &'a mut Vec<(FunctionId, TypeArguments<'ctx>)>,
        visited: &'a BTreeSet<(FunctionId, TypeArguments<'ctx>)>,
        function_id: FunctionId,
        type_arguments: TypeArguments<'ctx>,
    }

    impl<'a, 'ctx> ExprVisitor<'a, 'ctx> for CollectVisitor<'_, 'ctx> {
        fn visit_expr(&mut self, expr: &'a Expr<'ctx>) {
            expr.visit_children(self);

            if let ExprKind::FunctionCall(id, type_arguments, _) = &expr.kind {
                let raw_type_arguments: Vec<_> = type_arguments
                    .0
                    .get()
                    .iter()
                    .map(|ty| {
                        ty.instantiate(
                            self.ctx,
                            TypeParameterOwner::Function(self.function_id),
                            self.type_arguments,
                        )
                    })
                    .collect();
                let type_arguments = TypeArguments::new(self.ctx, &raw_type_arguments);
                if !self.visited.contains(&(*id, type_arguments)) {
                    self.queue.push((*id, type_arguments));
                }
            }
        }
    }

    let mut queue = Vec::new();
    let mut retval = BTreeSet::new();
    let empty_type_args = TypeArguments::new(module.ctx, &[]);

    for (id, function) in &module.functions {
        if function.type_parameters == 0 {
            queue.push((*id, empty_type_args));
        }
    }

    while let Some((id, type_arguments)) = queue.pop() {
        if !retval.insert((id, type_arguments)) {
            continue;
        }

        if let Some(body) = &module.functions[&id].body {
            CollectVisitor {
                ctx: module.ctx,
                queue: &mut queue,
                visited: &retval,
                function_id: id,
                type_arguments,
            }
            .visit_expr(body);
        }
    }

    retval
}

pub fn monomorphize<'ctx>(module: &mut Module<'ctx>, entries: &BTreeSet<(FunctionId, TypeArguments<'ctx>)>) {
    let mut monomorphized_cache = HashMap::<(FunctionId, TypeArguments<'ctx>), FunctionId>::new();
    let mut counter = 0;

    for &(poly_id, type_arguments) in entries {
        if type_arguments.is_empty() {
            continue;
        }
        let mono_id = FunctionId::new();
        let poly_fn = &module.functions[&poly_id];
        monomorphized_cache.insert((poly_id, type_arguments), mono_id);
        module.functions.insert(
            mono_id,
            Function {
                id: mono_id,
                name_ident: poly_fn.name_ident.clone(),
                mangled_name: format!("{}__epl_mono_{}", poly_fn.mangled_name, {
                    counter += 1;
                    counter
                }),
                debug_name: format!("{}{}", poly_fn.debug_name, type_arguments.render()),
                monomorphized_for: Some((poly_id, type_arguments)),
                type_parameters: 0,
                args: poly_fn
                    .args
                    .iter()
                    .map(|arg| {
                        (
                            arg.0.clone(),
                            arg.1
                                .instantiate(module.ctx, TypeParameterOwner::Function(poly_id), type_arguments),
                        )
                    })
                    .collect(),
                return_ty: poly_fn.return_ty.instantiate(
                    module.ctx,
                    TypeParameterOwner::Function(poly_id),
                    type_arguments,
                ),
                is_variadic: poly_fn.is_variadic,
                is_pure: poly_fn.is_pure,
                body: None,
            },
        );
    }

    for (&(poly_id, type_arguments), &mono_id) in &monomorphized_cache {
        if let Some(body) = &module.functions[&poly_id].body {
            let mut instantiation_ctx = InstantiationCtx {
                ctx: module.ctx,
                poly_id,
                type_arguments,
            };
            let mut body = body.clone();
            instantiation_ctx.visit_expr(&mut body);
            module.functions.get_mut(&mono_id).unwrap().body = Some(body);
        }
    }

    module
        .functions
        .retain(|_, f| f.type_parameters == 0 || f.monomorphized_for.is_some());

    for function in module.functions.values_mut() {
        if let Some(body) = &mut function.body {
            struct Specializer<'a, 'ctx> {
                ctx: Context<'ctx>,
                monomorphized_cache: &'a HashMap<(FunctionId, TypeArguments<'ctx>), FunctionId>,
            }
            impl<'a, 'ctx> ExprVisitorMut<'a, 'ctx> for Specializer<'_, 'ctx> {
                fn visit_expr(&mut self, expr: &'a mut Expr<'ctx>) {
                    expr.visit_children_mut(self);
                    if let ExprKind::FunctionCall(id, type_arguments, _) = &mut expr.kind
                        && let Some(mono_id) = self.monomorphized_cache.get(&(*id, *type_arguments))
                    {
                        *id = *mono_id;
                        *type_arguments = TypeArguments::new(self.ctx, &[]);
                    }
                }
            }
            let mut specializer = Specializer {
                ctx: module.ctx,
                monomorphized_cache: &monomorphized_cache,
            };
            specializer.visit_expr(body);
        }
    }
}

struct InstantiationCtx<'ctx> {
    ctx: Context<'ctx>,
    poly_id: FunctionId,
    type_arguments: TypeArguments<'ctx>,
}

impl<'ctx> InstantiationCtx<'ctx> {
    fn visit_type(&mut self, ty: &mut Type<'ctx>) {
        *ty = ty.instantiate(
            self.ctx,
            TypeParameterOwner::Function(self.poly_id),
            self.type_arguments,
        );
    }
}

impl<'a, 'ctx> ExprVisitorMut<'a, 'ctx> for InstantiationCtx<'ctx> {
    fn visit_expr(&mut self, expr: &'a mut Expr<'ctx>) {
        expr.visit_children_mut(self);
        self.visit_type(&mut expr.ty);

        match &mut expr.kind {
            ExprKind::Block(blk) => {
                for var in &mut blk.variables {
                    self.visit_type(&mut var.ty);
                }
            }
            ExprKind::FunctionCall(_, type_arguments, _) => {
                let raw_type_arguments: Vec<_> = type_arguments
                    .0
                    .get()
                    .iter()
                    .map(|ty| {
                        ty.instantiate(
                            self.ctx,
                            TypeParameterOwner::Function(self.poly_id),
                            self.type_arguments,
                        )
                    })
                    .collect();
                *type_arguments = TypeArguments::new(self.ctx, &raw_type_arguments);
            }
            ExprKind::Const(_)
            | ExprKind::ConstString(_)
            | ExprKind::Load(_)
            | ExprKind::Field(_, _)
            | ExprKind::ArrayElement(_, _)
            | ExprKind::Store(_, _)
            | ExprKind::GetPointer(_)
            | ExprKind::Argument(_)
            | ExprKind::Return(_)
            | ExprKind::Break(_, _)
            | ExprKind::Continue(_)
            | ExprKind::Arithmetic(_, _, _)
            | ExprKind::InPlaceArithmetic(_, _, _)
            | ExprKind::Cmp(_, _, _)
            | ExprKind::If { .. }
            | ExprKind::Loop(_, _)
            | ExprKind::ArrayInitializer(_)
            | ExprKind::StructInitializer(_)
            | ExprKind::Cast(_)
            | ExprKind::Not(_)
            | ExprKind::Comptime(_) => (),
        }
    }

    fn visit_place(&mut self, place: &'a mut Place<'ctx>) {
        place.visit_children_mut(self);
        self.visit_type(&mut place.ty);
    }
}
