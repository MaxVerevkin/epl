use std::collections::HashMap;

use super::*;
use crate::ast;

/// Construct an IR of a function from its AST
pub fn build_function(
    decl: &FunctionDecl,
    body: &ast::BlockExpr,
    function_decls: &HashMap<String, FunctionDecl>,
    typesystem: &TypeSystem,
    type_namespace: &HashMap<String, Type>,
) -> Result<Function, Error> {
    if decl.is_variadic {
        return Err(
            Error::new("defining variadic functions is not supported").with_span(decl.name.span)
        );
    }

    let mut builder =
        FunctionBuilder::new(decl.return_ty, function_decls, typesystem, type_namespace);

    let mut entry_block_args = Vec::new();
    for arg in &decl.args {
        let alloca = builder.alloca(arg.ty);
        let block_arg = DefinitionId::new(arg.ty);
        entry_block_args.push(block_arg);
        builder
            .scope
            .variables
            .insert(arg.name.value.clone(), (alloca, arg.ty));
        builder
            .cursor()
            .store(Value::Definition(alloca), Value::Definition(block_arg));
    }

    let entry = builder.current_block_id;
    let body_eval = builder.eval_block_expr(body, Some(decl.return_ty))?;
    builder.finalize_block(match body_eval.value {
        MaybeValue::Diverges => Terminator::Unreachable,
        MaybeValue::Value(value) => Terminator::Return { value },
    });
    builder.basic_blocks.get_mut(&entry).unwrap().args = entry_block_args;

    Ok(Function {
        allocas: builder.allocas,
        entry,
        basic_blokcs: builder.basic_blocks,
    })
}

/// A function's IR builder context
struct FunctionBuilder<'a> {
    return_ty: Type,
    function_decls: &'a HashMap<String, FunctionDecl>,
    typesystem: &'a TypeSystem,
    type_namespace: &'a HashMap<String, Type>,
    allocas: Vec<Alloca>,
    basic_blocks: HashMap<BasicBlockId, BasicBlock>,
    current_block_id: BasicBlockId,
    current_block_args: Vec<DefinitionId>,
    current_instructions: Vec<Instruction>,
    scope: Scope,
}

/// A lexical scope
#[derive(Default)]
struct Scope {
    variables: HashMap<String, (DefinitionId, Type)>,
    loop_context: Option<LoopContext>,
    parent: Option<Box<Self>>,
}

#[derive(Clone, Copy)]
struct LoopContext {
    break_to: BasicBlockId,
    break_used: bool,
}

impl Scope {
    /// Create a nested scope
    fn push(&mut self) {
        let parent = std::mem::take(self);
        self.parent = Some(Box::new(parent));
    }

    /// Pop the latest nested scope
    fn pop(&mut self) {
        *self = *self.parent.take().unwrap();
    }

    /// Lookup a variable by its name, recursively traversind the list of scopes
    fn lookup_variable(&self, name: &str) -> Option<(DefinitionId, Type)> {
        if let Some(definition_id) = self.variables.get(name) {
            return Some(*definition_id);
        }
        if let Some(parent) = &self.parent {
            return parent.lookup_variable(name);
        }
        None
    }

    /// Recursively lookup a loop context
    fn loop_context(&mut self) -> Option<&mut LoopContext> {
        if let Some(ctx) = &mut self.loop_context {
            return Some(ctx);
        }
        if let Some(parent) = &mut self.parent {
            return parent.loop_context();
        }
        None
    }
}

/// The result of an expression evaluation
#[derive(Debug)]
struct EvalResult {
    ty: Type,
    value: MaybeValue,
}

impl EvalResult {
    /// The result of expression evaluating to void
    const VOID: Self = Self {
        ty: Type::Void,
        value: MaybeValue::Value(Value::Constant(Constant::Void)),
    };
}

/// The value of an exprission, possibly missing due to the expression being diverging
#[derive(Debug)]
enum MaybeValue {
    Diverges,
    Value(Value),
}

impl<'a> FunctionBuilder<'a> {
    /// Create a new builder context
    fn new(
        return_ty: Type,
        function_decls: &'a HashMap<String, FunctionDecl>,
        typesystem: &'a TypeSystem,
        type_namespace: &'a HashMap<String, Type>,
    ) -> Self {
        Self {
            return_ty,
            function_decls,
            typesystem,
            type_namespace,
            allocas: Vec::new(),
            basic_blocks: HashMap::new(),
            current_block_id: BasicBlockId::new(),
            current_block_args: Vec::new(),
            current_instructions: Vec::new(),
            scope: Scope::default(),
        }
    }

    /// Finalize the current basic block, and start editing a new empty basic block
    fn finalize_block(&mut self, terminator: Terminator) {
        let instructions = std::mem::take(&mut self.current_instructions);
        let args = std::mem::take(&mut self.current_block_args);
        self.basic_blocks.insert(
            self.current_block_id,
            BasicBlock {
                args,
                instructions,
                terminator,
            },
        );
        self.current_block_id = BasicBlockId::new();
    }

    /// Get the cursor for the current basic block
    fn cursor(&mut self) -> InstructionCursor<'_> {
        InstructionCursor {
            buf: &mut self.current_instructions,
        }
    }

    /// Returns a new static allocation slot
    fn alloca(&mut self, ty: Type) -> DefinitionId {
        let alloca = DefinitionId::new(Type::OpaquePointer);
        self.allocas.push(Alloca {
            definition_id: alloca,
            layout: self.typesystem.layout_of(ty),
        });
        alloca
    }

    /// Evaluate an expression
    fn eval_expr(
        &mut self,
        expr: &ast::Expr,
        expect_type: Option<Type>,
    ) -> Result<EvalResult, Error> {
        match expr {
            ast::Expr::WithNoBlock(expr) => self.eval_expr_with_no_block(expr, expect_type),
            ast::Expr::WithBlock(expr) => self.eval_expr_with_block(expr, expect_type),
        }
    }

    /// Evaluate a place expression, the value returned is the pointer, type is pointee type.
    fn eval_place_expr(&mut self, expr: &ast::Expr) -> Result<EvalResult, Error> {
        match expr {
            ast::Expr::WithNoBlock(expr) => self.eval_place_expr_with_no_block(expr),
            _ => Err(Error::new("expected a place expression (ident)").with_span(expr.span())),
        }
    }

    /// Evaluate a place expression, the value returned is the pointer, type is pointee type.
    fn eval_place_expr_with_no_block(
        &mut self,
        expr: &ast::ExprWithNoBlock,
    ) -> Result<EvalResult, Error> {
        match expr {
            ast::ExprWithNoBlock::Ident(ident) => match self.scope.lookup_variable(&ident.value) {
                Some((alloca, ty)) => Ok(EvalResult {
                    ty,
                    value: MaybeValue::Value(Value::Definition(alloca)),
                }),
                None => Err(Error::new(format!("variable {:?} not found", ident.value))
                    .with_span(ident.span)),
            },
            ast::ExprWithNoBlock::FieldAccess(field_access_expr) => {
                let lhs_place = self.eval_place_expr(&field_access_expr.lhs)?;
                let sid = match lhs_place.ty {
                    Type::Struct(sid) => sid,
                    other => {
                        return Err(Error::new(format!(
                            "only structs can be field-accessed, not {other:?}"
                        ))
                        .with_span(field_access_expr.dot_span));
                    }
                };
                let (offset, field_ty) = self
                    .typesystem
                    .get_struct_field(sid, &field_access_expr.field)?;
                match lhs_place.value {
                    MaybeValue::Diverges => Ok(EvalResult {
                        ty: field_ty,
                        value: MaybeValue::Diverges,
                    }),
                    MaybeValue::Value(value) => {
                        let ptr = match offset {
                            0 => value,
                            _ => Value::Definition(
                                self.cursor().offset_ptr(value, offset.try_into().unwrap()),
                            ),
                        };
                        Ok(EvalResult {
                            ty: field_ty,
                            value: MaybeValue::Value(ptr),
                        })
                    }
                }
            }
            _ => Err(Error::new("expected a place expression (ident)").with_span(expr.span())),
        }
    }

    /// Evaluate a block expression
    fn eval_block_expr(
        &mut self,
        expr: &ast::BlockExpr,
        expect_type: Option<Type>,
    ) -> Result<EvalResult, Error> {
        self.scope.push();
        let mut value = None;

        for (i, stmt) in expr.statements.iter().enumerate() {
            match stmt {
                ast::Statement::Empty => (),
                ast::Statement::Let(let_statement) => match let_statement {
                    ast::LetStatement::WithValue { name, ty, value } => {
                        let ty = ty
                            .as_ref()
                            .map(|ty| self.typesystem.type_from_ast(self.type_namespace, ty))
                            .transpose()?;
                        let value_eval = self.eval_expr(value, ty)?;
                        let alloca = self.alloca(value_eval.ty);
                        self.scope
                            .variables
                            .insert(name.value.clone(), (alloca, value_eval.ty));
                        match value_eval.value {
                            MaybeValue::Diverges => (),
                            MaybeValue::Value(value) => {
                                self.cursor().store(Value::Definition(alloca), value);
                            }
                        }
                    }
                    ast::LetStatement::WithoutValue { name, ty } => {
                        let ty = self.typesystem.type_from_ast(self.type_namespace, ty)?;
                        let alloca = self.alloca(ty);
                        self.scope
                            .variables
                            .insert(name.value.clone(), (alloca, ty));
                    }
                },
                ast::Statement::ExprWithNoBlock(expr_with_no_block) => {
                    self.eval_expr_with_no_block(expr_with_no_block, None)?;
                }
                ast::Statement::ExprWithBlock(expr_with_block) => {
                    if i + 1 == expr.statements.len() && expr.final_expr.is_none() {
                        value = Some(self.eval_expr_with_block(expr_with_block, expect_type)?);
                    } else {
                        self.eval_expr_with_block(expr_with_block, Some(Type::Void))?;
                    }
                }
            }
        }

        if let Some(final_expr) = &expr.final_expr {
            value = Some(self.eval_expr_with_no_block(final_expr, expect_type)?);
        }

        self.scope.pop();

        Ok(match value {
            Some(value) => value,
            None => {
                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::new(format!(
                        "expectd expr of type {expect_type:?}, found end-of-block"
                    ))
                    .with_span(expr.closing_brace_span));
                }
                EvalResult::VOID
            }
        })
    }

    /// Evaluate an expression with no block
    fn eval_expr_with_no_block(
        &mut self,
        expr: &ast::ExprWithNoBlock,
        expect_type: Option<Type>,
    ) -> Result<EvalResult, Error> {
        match expr {
            ast::ExprWithNoBlock::Return(return_expr) => {
                let value_eval = self.eval_expr(&return_expr.value, Some(self.return_ty))?;
                match value_eval.value {
                    MaybeValue::Diverges => (),
                    MaybeValue::Value(value) => self.finalize_block(Terminator::Return { value }),
                }
                Ok(EvalResult {
                    ty: Type::Never,
                    value: MaybeValue::Diverges,
                })
            }
            ast::ExprWithNoBlock::Break(break_expr) => match &break_expr.value {
                Some(value) => {
                    let value_eval = self.eval_expr(value, Some(Type::Void))?;
                    match value_eval.value {
                        MaybeValue::Diverges => (),
                        MaybeValue::Value(_value) => {
                            let loop_ctx = self.scope.loop_context().ok_or_else(|| {
                                Error::new("break expressions are only allowed inside loops")
                                    .with_span(break_expr.break_keyword_span)
                            })?;
                            loop_ctx.break_used = true;
                            let to = loop_ctx.break_to;
                            self.finalize_block(Terminator::Jump {
                                to,
                                args: Vec::new(),
                            });
                        }
                    }
                    Ok(EvalResult {
                        ty: Type::Never,
                        value: MaybeValue::Diverges,
                    })
                }
                None => {
                    let loop_ctx = self.scope.loop_context().ok_or_else(|| {
                        Error::new("break expressions are only allowed inside loops")
                            .with_span(break_expr.break_keyword_span)
                    })?;
                    loop_ctx.break_used = true;
                    let to = loop_ctx.break_to;
                    self.finalize_block(Terminator::Jump {
                        to,
                        args: Vec::new(),
                    });
                    Ok(EvalResult {
                        ty: Type::Never,
                        value: MaybeValue::Diverges,
                    })
                }
            },
            ast::ExprWithNoBlock::Literal(literal_expr) => match &literal_expr.value {
                ast::LiteralExprValue::Number(number) => {
                    if let Some(expect_type) = expect_type
                        && !expect_type.is_int()
                    {
                        Err(Error::expr_type_missmatch(
                            expect_type,
                            Type::CStr,
                            literal_expr.span,
                        ))
                    } else {
                        let ty = expect_type.unwrap_or(Type::I32);
                        let (bits, signed) = match ty {
                            Type::I32 => (32, true),
                            Type::U32 => (32, false),
                            Type::Never
                            | Type::Void
                            | Type::Bool
                            | Type::CStr
                            | Type::OpaquePointer
                            | Type::Struct(_) => {
                                unreachable!()
                            }
                        };
                        Ok(EvalResult {
                            ty,
                            value: MaybeValue::Value(Value::Constant(Constant::Number {
                                data: *number,
                                bits,
                                signed,
                            })),
                        })
                    }
                }
                ast::LiteralExprValue::String(string) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::CStr
                    {
                        Err(Error::expr_type_missmatch(
                            expect_type,
                            Type::CStr,
                            literal_expr.span,
                        ))
                    } else {
                        Ok(EvalResult {
                            ty: Type::CStr,
                            value: MaybeValue::Value(Value::Constant(Constant::String(
                                string.clone(),
                            ))),
                        })
                    }
                }
                ast::LiteralExprValue::Bool(bool) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        Err(Error::expr_type_missmatch(
                            expect_type,
                            Type::Bool,
                            literal_expr.span,
                        ))
                    } else {
                        Ok(EvalResult {
                            ty: Type::Bool,
                            value: MaybeValue::Value(Value::Constant(Constant::Bool(*bool))),
                        })
                    }
                }
            },
            ast::ExprWithNoBlock::FunctionCallExpr(function_call_expr) => {
                let decl = self
                    .function_decls
                    .get(&function_call_expr.name.value)
                    .ok_or_else(|| {
                        Error::new(format!(
                            "function {:?} not found",
                            function_call_expr.name.value
                        ))
                        .with_span(function_call_expr.name.span)
                    })?;
                let mut args_diverges = false;
                let mut args_values = Vec::new();
                if decl.is_variadic {
                    if decl.args.len() > function_call_expr.args.len() {
                        return Err(Error::new(format!(
                            "expected at least {} argument(s), found {}",
                            decl.args.len(),
                            function_call_expr.args.len()
                        ))
                        .with_span(function_call_expr.args_span));
                    }
                } else if decl.args.len() != function_call_expr.args.len() {
                    return Err(Error::new(format!(
                        "expected {} argument(s), found {}",
                        decl.args.len(),
                        function_call_expr.args.len()
                    ))
                    .with_span(function_call_expr.args_span));
                }
                for (arg_i, arg_expr) in function_call_expr.args.iter().enumerate() {
                    let expect_arg_type = decl.args.get(arg_i).map(|a| a.ty);
                    let arg_eval = self.eval_expr(arg_expr, expect_arg_type)?;
                    if !args_diverges {
                        match arg_eval.value {
                            MaybeValue::Diverges => args_diverges = true,
                            MaybeValue::Value(value) => args_values.push(value),
                        }
                    }
                }
                if let Some(expect_type) = expect_type
                    && expect_type != decl.return_ty
                    && decl.return_ty != Type::Never
                {
                    return Err(Error::expr_type_missmatch(
                        expect_type,
                        decl.return_ty,
                        function_call_expr.span(),
                    ));
                }
                if args_diverges {
                    Ok(EvalResult {
                        ty: decl.return_ty,
                        value: MaybeValue::Diverges,
                    })
                } else {
                    let definition_id = self.cursor().function_call(
                        function_call_expr.name.value.clone(),
                        args_values,
                        decl.return_ty,
                    );
                    if decl.return_ty == Type::Never {
                        self.finalize_block(Terminator::Unreachable);
                        Ok(EvalResult {
                            ty: decl.return_ty,
                            value: MaybeValue::Diverges,
                        })
                    } else {
                        Ok(EvalResult {
                            ty: decl.return_ty,
                            value: MaybeValue::Value(Value::Definition(definition_id)),
                        })
                    }
                }
            }
            ast::ExprWithNoBlock::Assignment(assignment_expr) => {
                let place_eval = self.eval_place_expr(&assignment_expr.place)?;
                let value_eval = self.eval_expr(&assignment_expr.value, Some(place_eval.ty))?;
                if let MaybeValue::Value(ptr) = place_eval.value
                    && let MaybeValue::Value(value) = value_eval.value
                {
                    self.cursor().store(ptr, value);
                }
                Ok(EvalResult::VOID)
            }
            ast::ExprWithNoBlock::AddAssignment(assignment_expr) => {
                let place_eval = self.eval_place_expr(&assignment_expr.place)?;
                if !place_eval.ty.is_int() {
                    return Err(Error::new(format!(
                        "only integer types can be added, not {:?}",
                        place_eval.ty
                    ))
                    .with_span(assignment_expr.op_span));
                }
                let value_eval = self.eval_expr(&assignment_expr.value, Some(place_eval.ty))?;
                if let MaybeValue::Value(ptr) = place_eval.value
                    && let MaybeValue::Value(value) = value_eval.value
                {
                    let lhs = self.cursor().load(ptr.clone(), place_eval.ty);
                    let result = self.cursor().add(Value::Definition(lhs), value);
                    self.cursor().store(ptr, Value::Definition(result));
                }
                Ok(EvalResult::VOID)
            }
            ast::ExprWithNoBlock::SubAssignment(_) => todo!(),
            ast::ExprWithNoBlock::MulAssignment(_) => todo!(),
            ast::ExprWithNoBlock::DivAssignment(_) => todo!(),
            ast::ExprWithNoBlock::Binary(binary_expr) => match binary_expr.op {
                ast::BinaryOp::Equal => todo!(),
                ast::BinaryOp::NotEqual => todo!(),
                ast::BinaryOp::LessOrEqual => todo!(),
                ast::BinaryOp::GreaterOrEqual => todo!(),
                ast::BinaryOp::Less => {
                    let lhs_eval = self.eval_expr(&binary_expr.lhs, None)?;
                    let rhs_eval = self.eval_expr(&binary_expr.rhs, Some(lhs_eval.ty))?;
                    if lhs_eval.ty != rhs_eval.ty {
                        return Err(Error::new(format!(
                            "cannot comparet different types: {:?} and {:?}",
                            lhs_eval.ty, rhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    if !lhs_eval.ty.is_int() {
                        return Err(Error::new(format!(
                            "only integer types can be compared, not {:?}",
                            lhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    Ok(EvalResult {
                        ty: Type::Bool,
                        value: if let MaybeValue::Value(lhs) = lhs_eval.value
                            && let MaybeValue::Value(rhs) = rhs_eval.value
                        {
                            MaybeValue::Value(Value::Definition(self.cursor().cmp_l(lhs, rhs)))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
                ast::BinaryOp::Greater => {
                    let lhs_eval = self.eval_expr(&binary_expr.lhs, None)?;
                    let rhs_eval = self.eval_expr(&binary_expr.rhs, Some(lhs_eval.ty))?;
                    if lhs_eval.ty != rhs_eval.ty {
                        return Err(Error::new(format!(
                            "cannot comparet different types: {:?} and {:?}",
                            lhs_eval.ty, rhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    if !lhs_eval.ty.is_int() {
                        return Err(Error::new(format!(
                            "only integer types can be compared, not {:?}",
                            lhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    Ok(EvalResult {
                        ty: Type::Bool,
                        value: if let MaybeValue::Value(lhs) = lhs_eval.value
                            && let MaybeValue::Value(rhs) = rhs_eval.value
                        {
                            MaybeValue::Value(Value::Definition(self.cursor().cmp_g(lhs, rhs)))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
                ast::BinaryOp::Add => {
                    let lhs_eval = self.eval_expr(&binary_expr.lhs, None)?;
                    let rhs_eval = self.eval_expr(&binary_expr.rhs, Some(lhs_eval.ty))?;
                    if lhs_eval.ty != rhs_eval.ty {
                        return Err(Error::new(format!(
                            "cannot add different types: {:?} and {:?}",
                            lhs_eval.ty, rhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    if !lhs_eval.ty.is_int() {
                        return Err(Error::new(format!(
                            "only integer types can be added, not {:?}",
                            lhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    Ok(EvalResult {
                        ty: lhs_eval.ty,
                        value: if let MaybeValue::Value(lhs) = lhs_eval.value
                            && let MaybeValue::Value(rhs) = rhs_eval.value
                        {
                            MaybeValue::Value(Value::Definition(self.cursor().add(lhs, rhs)))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
                ast::BinaryOp::Sub => {
                    let lhs_eval = self.eval_expr(&binary_expr.lhs, None)?;
                    let rhs_eval = self.eval_expr(&binary_expr.rhs, Some(lhs_eval.ty))?;
                    if lhs_eval.ty != rhs_eval.ty {
                        return Err(Error::new(format!(
                            "cannot subtract different types: {:?} and {:?}",
                            lhs_eval.ty, rhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    if !lhs_eval.ty.is_int() {
                        return Err(Error::new(format!(
                            "only integer types can be subtracted, not {:?}",
                            lhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    Ok(EvalResult {
                        ty: lhs_eval.ty,
                        value: if let MaybeValue::Value(lhs) = lhs_eval.value
                            && let MaybeValue::Value(rhs) = rhs_eval.value
                        {
                            MaybeValue::Value(Value::Definition(self.cursor().sub(lhs, rhs)))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
                ast::BinaryOp::Mul => {
                    let lhs_eval = self.eval_expr(&binary_expr.lhs, None)?;
                    let rhs_eval = self.eval_expr(&binary_expr.rhs, Some(lhs_eval.ty))?;
                    if lhs_eval.ty != rhs_eval.ty {
                        return Err(Error::new(format!(
                            "cannot multipy different types: {:?} and {:?}",
                            lhs_eval.ty, rhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    if !lhs_eval.ty.is_int() {
                        return Err(Error::new(format!(
                            "only integer types can be multiplied, not {:?}",
                            lhs_eval.ty
                        ))
                        .with_span(binary_expr.op_span));
                    }
                    Ok(EvalResult {
                        ty: lhs_eval.ty,
                        value: if let MaybeValue::Value(lhs) = lhs_eval.value
                            && let MaybeValue::Value(rhs) = rhs_eval.value
                        {
                            MaybeValue::Value(Value::Definition(self.cursor().mul(lhs, rhs)))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
                ast::BinaryOp::Div => todo!(),
            },
            ast::ExprWithNoBlock::Unary(unary_expr) => match unary_expr.op {
                ast::UnaryOp::Negate => {
                    let rhs_eval = self.eval_expr(&unary_expr.rhs, None)?;
                    if !rhs_eval.ty.is_signed_int() {
                        return Err(Error::new(format!(
                            "only signed integer types can be negated, not {:?}",
                            rhs_eval.ty
                        ))
                        .with_span(unary_expr.op_span));
                    }
                    let (bits, signed) = match rhs_eval.ty {
                        Type::I32 => (32, true),
                        Type::Never
                        | Type::Void
                        | Type::Bool
                        | Type::U32
                        | Type::CStr
                        | Type::OpaquePointer
                        | Type::Struct(_) => {
                            unreachable!()
                        }
                    };
                    Ok(EvalResult {
                        ty: rhs_eval.ty,
                        value: if let MaybeValue::Value(rhs) = rhs_eval.value {
                            MaybeValue::Value(Value::Definition(self.cursor().sub(
                                Value::Constant(Constant::Number {
                                    data: 0,
                                    bits,
                                    signed,
                                }),
                                rhs,
                            )))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
                ast::UnaryOp::Not => {
                    let rhs_eval = self.eval_expr(&unary_expr.rhs, Some(Type::Bool))?;
                    Ok(EvalResult {
                        ty: rhs_eval.ty,
                        value: if let MaybeValue::Value(rhs) = rhs_eval.value {
                            MaybeValue::Value(Value::Definition(self.cursor().not(rhs)))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
            },
            ast::ExprWithNoBlock::Ident(_) | ast::ExprWithNoBlock::FieldAccess(_) => {
                let place = self.eval_place_expr_with_no_block(expr)?;
                if let Some(expect_type) = expect_type
                    && expect_type != place.ty
                {
                    return Err(Error::expr_type_missmatch(
                        expect_type,
                        place.ty,
                        expr.span(),
                    ));
                }
                match place.value {
                    MaybeValue::Diverges => Ok(EvalResult {
                        ty: place.ty,
                        value: MaybeValue::Diverges,
                    }),
                    MaybeValue::Value(value) => {
                        let definition_id = self.cursor().load(value, place.ty);
                        Ok(EvalResult {
                            ty: place.ty,
                            value: MaybeValue::Value(Value::Definition(definition_id)),
                        })
                    }
                }
            }
        }
    }

    /// Evaluate an expression with block
    fn eval_expr_with_block(
        &mut self,
        expr: &ast::ExprWithBlock,
        expect_type: Option<Type>,
    ) -> Result<EvalResult, Error> {
        match expr {
            ast::ExprWithBlock::Block(block_expr) => self.eval_block_expr(block_expr, expect_type),
            ast::ExprWithBlock::If(if_expr) if if_expr.if_false.is_none() => {
                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::new(format!(
                        "if experession expected to evalueate to type {expect_type:?}, so it must have an else branch"
                    ))
                    .with_span(if_expr.if_keyword_span));
                }

                let continuation_id = BasicBlockId::new();
                let if_true_id = BasicBlockId::new();

                let cond = self.eval_expr(&if_expr.cond, Some(Type::Bool))?;
                self.finalize_block(match cond.value {
                    MaybeValue::Diverges => Terminator::Unreachable,
                    MaybeValue::Value(cond) => Terminator::CondJump {
                        cond,
                        if_true: if_true_id,
                        if_true_args: Vec::new(),
                        if_false: continuation_id,
                        if_false_args: Vec::new(),
                    },
                });

                self.current_block_id = if_true_id;
                self.eval_block_expr(&if_expr.if_true, Some(Type::Void))?;
                self.finalize_block(Terminator::Jump {
                    to: continuation_id,
                    args: Vec::new(),
                });
                self.current_block_id = continuation_id;

                Ok(EvalResult::VOID)
            }
            ast::ExprWithBlock::If(if_expr) => {
                let if_false_expr = if_expr.if_false.as_ref().unwrap();

                let continuation_id = BasicBlockId::new();
                let if_true_id = BasicBlockId::new();
                let if_false_id = BasicBlockId::new();

                let cond = self.eval_expr(&if_expr.cond, Some(Type::Bool))?;
                self.finalize_block(match cond.value {
                    MaybeValue::Diverges => Terminator::Unreachable,
                    MaybeValue::Value(cond) => Terminator::CondJump {
                        cond,
                        if_true: if_true_id,
                        if_true_args: Vec::new(),
                        if_false: if_false_id,
                        if_false_args: Vec::new(),
                    },
                });

                self.current_block_id = if_true_id;
                let if_true_eval = self.eval_block_expr(&if_expr.if_true, expect_type)?;
                self.finalize_block(match if_true_eval.value {
                    MaybeValue::Diverges => Terminator::Unreachable,
                    MaybeValue::Value(value) => Terminator::Jump {
                        to: continuation_id,
                        args: vec![value],
                    },
                });
                self.current_block_id = if_false_id;
                let if_false_eval = self.eval_block_expr(if_false_expr, expect_type)?;
                self.finalize_block(match if_false_eval.value {
                    MaybeValue::Diverges => Terminator::Unreachable,
                    MaybeValue::Value(value) => Terminator::Jump {
                        to: continuation_id,
                        args: vec![value],
                    },
                });
                self.current_block_id = continuation_id;

                let Some(result_ty) = if_true_eval.ty.comine_ignoring_never(if_false_eval.ty)
                else {
                    return Err(Error::new(format!(
                        "if expression branches evaluate to different types: {:?} and {:?}",
                        if_true_eval.ty, if_false_eval.ty
                    ))
                    .with_span(if_expr.if_keyword_span));
                };

                let value = DefinitionId::new(result_ty);
                self.current_block_args.push(value);
                Ok(EvalResult {
                    ty: result_ty,
                    value: MaybeValue::Value(Value::Definition(value)),
                })
            }
            ast::ExprWithBlock::Loop(loop_expr) => {
                let body_id = BasicBlockId::new();
                let continuation_id = BasicBlockId::new();

                self.finalize_block(Terminator::Jump {
                    to: body_id,
                    args: Vec::new(),
                });

                self.current_block_id = body_id;
                self.scope.push();
                self.scope.loop_context = Some(LoopContext {
                    break_to: continuation_id,
                    break_used: false,
                });
                let _body_eval = self.eval_block_expr(&loop_expr.body, Some(Type::Void))?;
                let loop_ctx = self.scope.loop_context.unwrap();
                self.scope.pop();
                self.finalize_block(Terminator::Jump {
                    to: body_id,
                    args: Vec::new(),
                });

                self.current_block_id = continuation_id;

                if loop_ctx.break_used {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Void
                    {
                        return Err(Error::new(format!(
                            "loop experession expected to evalueate to type {expect_type:?}, but break is used, so it evaluates to Void"
                        )).with_span(loop_expr.loop_keyword_span));
                    }
                    Ok(EvalResult::VOID)
                } else {
                    Ok(EvalResult {
                        ty: Type::Never,
                        value: MaybeValue::Diverges,
                    })
                }
            }
            ast::ExprWithBlock::While(while_expr) => {
                let header_id = BasicBlockId::new();
                let body_id = BasicBlockId::new();
                let continuation_id = BasicBlockId::new();

                self.finalize_block(Terminator::Jump {
                    to: header_id,
                    args: Vec::new(),
                });

                self.current_block_id = header_id;
                let cond_eval = self.eval_expr(&while_expr.cond, Some(Type::Bool))?;
                self.finalize_block(match cond_eval.value {
                    MaybeValue::Diverges => Terminator::Unreachable,
                    MaybeValue::Value(cond) => Terminator::CondJump {
                        cond,
                        if_true: body_id,
                        if_true_args: Vec::new(),
                        if_false: continuation_id,
                        if_false_args: Vec::new(),
                    },
                });

                self.current_block_id = body_id;
                self.scope.push();
                self.scope.loop_context = Some(LoopContext {
                    break_to: continuation_id,
                    break_used: false,
                });
                let _body_eval = self.eval_block_expr(&while_expr.body, Some(Type::Void))?;
                self.scope.pop();
                self.finalize_block(Terminator::Jump {
                    to: header_id,
                    args: Vec::new(),
                });

                self.current_block_id = continuation_id;

                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::expr_type_missmatch(
                        expect_type,
                        Type::Void,
                        while_expr.while_keyword_span,
                    ));
                }

                Ok(EvalResult::VOID)
            }
        }
    }
}

/// An instruction cursor used to append instructions at the end of a basic block
struct InstructionCursor<'a> {
    buf: &'a mut Vec<Instruction>,
}

impl InstructionCursor<'_> {
    /// Generate a `Load` instruction
    fn load(&mut self, ptr: Value, ty: Type) -> DefinitionId {
        let definition_id = DefinitionId::new(ty);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::Load { ptr },
        });
        definition_id
    }

    /// Generate a `Store` instruction
    fn store(&mut self, ptr: Value, value: Value) {
        self.buf.push(Instruction {
            definition_id: DefinitionId::new(Type::Void),
            kind: InstructionKind::Store { ptr, value },
        });
    }

    /// Generate a `FunctionCall` instruction
    fn function_call(&mut self, name: String, args: Vec<Value>, ty: Type) -> DefinitionId {
        let definition_id = DefinitionId::new(ty);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::FunctionCall { name, args },
        });
        definition_id
    }

    /// Generate a `CmpL` instruction
    fn cmp_l(&mut self, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::CmpL { lhs, rhs },
        });
        definition_id
    }

    /// Generate a `CmpG` instruction
    fn cmp_g(&mut self, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::CmpG { lhs, rhs },
        });
        definition_id
    }

    /// Generate a `Add` instruction
    fn add(&mut self, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(lhs.ty());
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::Add { lhs, rhs },
        });
        definition_id
    }

    /// Generate a `Sub` instruction
    fn sub(&mut self, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(lhs.ty());
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::Sub { lhs, rhs },
        });
        definition_id
    }

    /// Generate a `Mul` instruction
    fn mul(&mut self, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(lhs.ty());
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::Mul { lhs, rhs },
        });
        definition_id
    }

    /// Generate a `Not` instruction
    fn not(&mut self, value: Value) -> DefinitionId {
        assert_eq!(value.ty(), Type::Bool);
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::Not { value },
        });
        definition_id
    }

    /// Generate a `OffsetPtr` instruction
    fn offset_ptr(&mut self, ptr: Value, offset: i64) -> DefinitionId {
        let definition_id = DefinitionId::new(Type::OpaquePointer);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::OffsetPtr { ptr, offset },
        });
        definition_id
    }
}
