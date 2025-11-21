use std::collections::HashMap;

use super::*;
use crate::ast;

/// Construct an IR of a function from its AST
pub fn build_function(
    decl: &FunctionDecl,
    body: &ast::BlockExpr,
    function_decls: &HashMap<String, FunctionDecl>,
    typesystem: &mut TypeSystem,
    type_namespace: &HashMap<String, Type>,
) -> Result<Function, Error> {
    if decl.is_variadic {
        return Err(Error::new("defining variadic functions is not supported").with_span(decl.name.span));
    }

    let mut builder = FunctionBuilder::new(decl.return_ty, function_decls, typesystem, type_namespace);

    let mut entry_block_args = Vec::new();
    for arg in &decl.args {
        let alloca = builder.alloca(arg.ty);
        let block_arg = DefinitionId::new(arg.ty);
        entry_block_args.push(block_arg);
        builder.scope.variables.insert(arg.name.value.clone(), (alloca, arg.ty));
        builder
            .cursor()
            .store(Value::Definition(alloca), Value::Definition(block_arg));
    }

    let entry = builder.current_block_id;
    let body_eval = builder.eval_block_expr(body, Some(decl.return_ty))?;
    builder.finalize_block(match body_eval {
        EvalResult::Diverges(_) => Terminator::Unreachable,
        EvalResult::Value(value) => Terminator::Return { value },
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
    typesystem: &'a mut TypeSystem,
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

/// The value of an exprission, possibly missing due to the expression being diverging
#[derive(Debug)]
enum EvalResult {
    Diverges(Type),
    Value(Value),
}

impl EvalResult {
    /// The result of expression evaluating to void
    const VOID: Self = Self::Value(Value::Constant(Constant::Void));

    /// The type of the expression
    fn ty(&self) -> Type {
        match self {
            Self::Diverges(ty) => *ty,
            Self::Value(value) => value.ty(),
        }
    }
}

impl<'a> FunctionBuilder<'a> {
    /// Create a new builder context
    fn new(
        return_ty: Type,
        function_decls: &'a HashMap<String, FunctionDecl>,
        typesystem: &'a mut TypeSystem,
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
    fn eval_expr(&mut self, expr: &ast::Expr, expect_type: Option<Type>) -> Result<EvalResult, Error> {
        match expr {
            ast::Expr::WithNoBlock(expr) => self.eval_expr_with_no_block(expr, expect_type),
            ast::Expr::WithBlock(expr) => self.eval_expr_with_block(expr, expect_type),
        }
    }

    /// Evaluate a place expression, the value returned is the pointer, type is pointee type.
    fn eval_place_expr(&mut self, expr: &ast::Expr) -> Result<(EvalResult, Type), Error> {
        match expr {
            ast::Expr::WithNoBlock(expr) => self.eval_place_expr_with_no_block(expr),
            _ => Err(Error::new("expected a place expression (ident)").with_span(expr.span())),
        }
    }

    /// Evaluate a place expression, the value returned is the pointer, type is pointee type.
    fn eval_place_expr_with_no_block(&mut self, expr: &ast::ExprWithNoBlock) -> Result<(EvalResult, Type), Error> {
        match expr {
            ast::ExprWithNoBlock::Ident(ident) => match self.scope.lookup_variable(&ident.value) {
                Some((alloca, ty)) => Ok((EvalResult::Value(Value::Definition(alloca)), ty)),
                None => Err(Error::new(format!("variable {:?} not found", ident.value)).with_span(ident.span)),
            },
            ast::ExprWithNoBlock::FieldAccess(field_access_expr) => {
                let (lhs_place, lhs_place_ty) = self.eval_place_expr(&field_access_expr.lhs)?;
                let sid = match lhs_place_ty {
                    Type::Struct(sid) => sid,
                    other => {
                        return Err(Error::new(format!("only structs can be field-accessed, not {other:?}"))
                            .with_span(field_access_expr.dot_span));
                    }
                };
                let (offset, field_ty) = self.typesystem.get_struct_field(sid, &field_access_expr.field)?;
                match lhs_place {
                    EvalResult::Diverges(_) => Ok((EvalResult::Diverges(Type::Never), field_ty)),
                    EvalResult::Value(value) => {
                        let ptr = self.cursor().offset_ptr(value, offset.try_into().unwrap());
                        Ok((EvalResult::Value(ptr), field_ty))
                    }
                }
            }
            _ => Err(Error::new("expected a place expression (ident)").with_span(expr.span())),
        }
    }

    /// Evaluate a block expression
    fn eval_block_expr(&mut self, expr: &ast::BlockExpr, expect_type: Option<Type>) -> Result<EvalResult, Error> {
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
                        let value_ty = value_eval.ty();
                        let alloca = self.alloca(value_ty);
                        self.scope.variables.insert(name.value.clone(), (alloca, value_ty));
                        match value_eval {
                            EvalResult::Diverges(_) => (),
                            EvalResult::Value(value) => self.cursor().store(Value::Definition(alloca), value),
                        }
                    }
                    ast::LetStatement::WithoutValue { name, ty } => {
                        let ty = self.typesystem.type_from_ast(self.type_namespace, ty)?;
                        let alloca = self.alloca(ty);
                        self.scope.variables.insert(name.value.clone(), (alloca, ty));
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
                    return Err(
                        Error::new(format!("expectd expr of type {expect_type:?}, found end-of-block"))
                            .with_span(expr.closing_brace_span),
                    );
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
                match self.eval_expr(&return_expr.value, Some(self.return_ty))? {
                    EvalResult::Diverges(_) => (),
                    EvalResult::Value(value) => self.finalize_block(Terminator::Return { value }),
                }
                Ok(EvalResult::Diverges(Type::Never))
            }
            ast::ExprWithNoBlock::Break(break_expr) => match &break_expr.value {
                Some(value) => {
                    match self.eval_expr(value, Some(Type::Void))? {
                        EvalResult::Diverges(_) => (),
                        EvalResult::Value(_) => {
                            let loop_ctx = self.scope.loop_context().ok_or_else(|| {
                                Error::new("break expressions are only allowed inside loops")
                                    .with_span(break_expr.break_keyword_span)
                            })?;
                            loop_ctx.break_used = true;
                            let to = loop_ctx.break_to;
                            self.finalize_block(Terminator::Jump { to, args: Vec::new() });
                        }
                    }
                    Ok(EvalResult::Diverges(Type::Never))
                }
                None => {
                    let loop_ctx = self.scope.loop_context().ok_or_else(|| {
                        Error::new("break expressions are only allowed inside loops")
                            .with_span(break_expr.break_keyword_span)
                    })?;
                    loop_ctx.break_used = true;
                    let to = loop_ctx.break_to;
                    self.finalize_block(Terminator::Jump { to, args: Vec::new() });
                    Ok(EvalResult::Diverges(Type::Never))
                }
            },
            ast::ExprWithNoBlock::Literal(literal_expr) => match &literal_expr.value {
                ast::LiteralExprValue::Number(number) => {
                    if let Some(expect_type) = expect_type
                        && !expect_type.is_int()
                    {
                        Err(Error::expr_type_missmatch(expect_type, Type::CStr, literal_expr.span))
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
                            | Type::Ptr(_)
                            | Type::Struct(_) => {
                                unreachable!()
                            }
                        };
                        Ok(EvalResult::Value(Value::Constant(Constant::Number {
                            data: *number,
                            bits,
                            signed,
                        })))
                    }
                }
                ast::LiteralExprValue::String(string) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::CStr
                    {
                        Err(Error::expr_type_missmatch(expect_type, Type::CStr, literal_expr.span))
                    } else {
                        Ok(EvalResult::Value(Value::Constant(Constant::String(string.clone()))))
                    }
                }
                ast::LiteralExprValue::Bool(bool) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        Err(Error::expr_type_missmatch(expect_type, Type::Bool, literal_expr.span))
                    } else {
                        Ok(EvalResult::Value(Value::Constant(Constant::Bool(*bool))))
                    }
                }
                ast::LiteralExprValue::Undefined => {
                    let expect_type = expect_type
                        .ok_or_else(|| Error::new("type annotations needed").with_span(literal_expr.span))?;
                    Ok(EvalResult::Value(Value::Constant(Constant::Undefined(expect_type))))
                }
            },
            ast::ExprWithNoBlock::FunctionCallExpr(function_call_expr) => {
                let decl = self.function_decls.get(&function_call_expr.name.value).ok_or_else(|| {
                    Error::new(format!("function {:?} not found", function_call_expr.name.value))
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
                        match arg_eval {
                            EvalResult::Diverges(_) => args_diverges = true,
                            EvalResult::Value(value) => args_values.push(value),
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
                    Ok(EvalResult::Diverges(decl.return_ty))
                } else {
                    let definition_id =
                        self.cursor()
                            .function_call(function_call_expr.name.value.clone(), args_values, decl.return_ty);
                    if decl.return_ty == Type::Never {
                        self.finalize_block(Terminator::Unreachable);
                        Ok(EvalResult::Diverges(decl.return_ty))
                    } else {
                        Ok(EvalResult::Value(Value::Definition(definition_id)))
                    }
                }
            }
            ast::ExprWithNoBlock::Assignment(assignment_expr) => {
                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::expr_type_missmatch(expect_type, Type::Void, expr.span()));
                }
                let (place_eval, place_ty) = self.eval_place_expr(&assignment_expr.place)?;
                let value_eval = self.eval_expr(&assignment_expr.value, Some(place_ty))?;
                if let EvalResult::Value(ptr) = place_eval
                    && let EvalResult::Value(value) = value_eval
                {
                    self.cursor().store(ptr, value);
                }
                Ok(EvalResult::VOID)
            }
            ast::ExprWithNoBlock::CompoundAssignment(compound_assignment_expr) => {
                if let Some(expect_type) = expect_type
                    && expect_type != Type::Void
                {
                    return Err(Error::expr_type_missmatch(expect_type, Type::Void, expr.span()));
                }
                let (place_eval, place_ty) = self.eval_place_expr(&compound_assignment_expr.place)?;
                if !place_ty.is_int() {
                    return Err(
                        Error::new(format!("arithmetic is only supported for integers, not {place_ty:?}"))
                            .with_span(compound_assignment_expr.op_span),
                    );
                }
                let value_eval = self.eval_expr(&compound_assignment_expr.value, Some(place_ty))?;
                if let EvalResult::Value(ptr) = place_eval
                    && let EvalResult::Value(value) = value_eval
                {
                    let lhs = self.cursor().load(ptr.clone(), place_ty);
                    let result = self
                        .cursor()
                        .arithmetic(compound_assignment_expr.op, Value::Definition(lhs), value);
                    self.cursor().store(ptr, Value::Definition(result));
                }
                Ok(EvalResult::VOID)
            }
            ast::ExprWithNoBlock::Binary(binary_expr) => match binary_expr.op {
                ast::BinaryOp::Cmp(cmp_op) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        return Err(Error::expr_type_missmatch(expect_type, Type::Bool, expr.span()));
                    }
                    let lhs_eval = self.eval_expr(&binary_expr.lhs, None)?;
                    let rhs_eval = self.eval_expr(&binary_expr.rhs, Some(lhs_eval.ty()))?;
                    let Some(ty) = lhs_eval.ty().comine_ignoring_never(rhs_eval.ty()) else {
                        return Err(Error::new(format!(
                            "cannot compare different types: {:?} and {:?}",
                            lhs_eval.ty(),
                            rhs_eval.ty(),
                        ))
                        .with_span(binary_expr.op_span));
                    };
                    if !ty.is_int() {
                        return Err(Error::new(format!("only integer types can be compared, not {ty:?}"))
                            .with_span(binary_expr.op_span));
                    }
                    Ok(
                        if let EvalResult::Value(lhs) = lhs_eval
                            && let EvalResult::Value(rhs) = rhs_eval
                        {
                            EvalResult::Value(Value::Definition(self.cursor().cmp(cmp_op, lhs, rhs)))
                        } else {
                            EvalResult::Diverges(Type::Bool)
                        },
                    )
                }
                ast::BinaryOp::Arithmetic(arithmetic_op) => {
                    let lhs_eval = self.eval_expr(&binary_expr.lhs, None)?;
                    let rhs_eval = self.eval_expr(&binary_expr.rhs, Some(lhs_eval.ty()))?;
                    let Some(ty) = lhs_eval.ty().comine_ignoring_never(rhs_eval.ty()) else {
                        return Err(Error::new(format!(
                            "cannot perform arithmetic on different types: {:?} and {:?}",
                            lhs_eval.ty(),
                            rhs_eval.ty(),
                        ))
                        .with_span(binary_expr.op_span));
                    };
                    if !ty.is_int() {
                        return Err(
                            Error::new(format!("arithemitc can only be performed on integers, not {ty:?}"))
                                .with_span(binary_expr.op_span),
                        );
                    }
                    if let Some(expect_type) = expect_type
                        && expect_type != ty
                    {
                        return Err(Error::expr_type_missmatch(expect_type, ty, expr.span()));
                    }
                    Ok(
                        if let EvalResult::Value(lhs) = lhs_eval
                            && let EvalResult::Value(rhs) = rhs_eval
                        {
                            EvalResult::Value(Value::Definition(self.cursor().arithmetic(arithmetic_op, lhs, rhs)))
                        } else {
                            EvalResult::Diverges(ty)
                        },
                    )
                }
            },
            ast::ExprWithNoBlock::Unary(unary_expr) => match unary_expr.op {
                ast::UnaryOp::Negate => {
                    let rhs_eval = self.eval_expr(&unary_expr.rhs, None)?;
                    let ty = rhs_eval.ty();
                    if !ty.is_signed_int() {
                        return Err(
                            Error::new(format!("only signed integer types can be negated, not {ty:?}"))
                                .with_span(unary_expr.op_span),
                        );
                    }
                    if let Some(expect_type) = expect_type
                        && expect_type != ty
                    {
                        return Err(Error::expr_type_missmatch(expect_type, ty, expr.span()));
                    }
                    let (bits, signed) = match ty {
                        Type::I32 => (32, true),
                        Type::Never
                        | Type::Void
                        | Type::Bool
                        | Type::U32
                        | Type::CStr
                        | Type::OpaquePointer
                        | Type::Ptr(_)
                        | Type::Struct(_) => {
                            unreachable!()
                        }
                    };
                    Ok(if let EvalResult::Value(rhs) = rhs_eval {
                        EvalResult::Value(Value::Definition(self.cursor().arithmetic(
                            ArithmeticOp::Sub,
                            Value::Constant(Constant::Number { data: 0, bits, signed }),
                            rhs,
                        )))
                    } else {
                        EvalResult::Diverges(ty)
                    })
                }
                ast::UnaryOp::Not => {
                    let rhs_eval = self.eval_expr(&unary_expr.rhs, Some(Type::Bool))?;
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Bool
                    {
                        return Err(Error::expr_type_missmatch(expect_type, Type::Bool, expr.span()));
                    }
                    Ok(if let EvalResult::Value(rhs) = rhs_eval {
                        EvalResult::Value(Value::Definition(self.cursor().not(rhs)))
                    } else {
                        EvalResult::Diverges(Type::Bool)
                    })
                }
            },
            ast::ExprWithNoBlock::Ident(_) | ast::ExprWithNoBlock::FieldAccess(_) => {
                let (place_eval, place_ty) = self.eval_place_expr_with_no_block(expr)?;
                if let Some(expect_type) = expect_type
                    && expect_type != place_ty
                {
                    return Err(Error::expr_type_missmatch(expect_type, place_ty, expr.span()));
                }
                Ok(if let EvalResult::Value(ptr) = place_eval {
                    EvalResult::Value(Value::Definition(self.cursor().load(ptr, place_ty)))
                } else {
                    EvalResult::Diverges(place_ty)
                })
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

                let cond_eval = self.eval_expr(&if_expr.cond, Some(Type::Bool))?;
                self.finalize_block(match cond_eval {
                    EvalResult::Diverges(_) => Terminator::Unreachable,
                    EvalResult::Value(cond) => Terminator::CondJump {
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

                let cond_eval = self.eval_expr(&if_expr.cond, Some(Type::Bool))?;
                self.finalize_block(match cond_eval {
                    EvalResult::Diverges(_) => Terminator::Unreachable,
                    EvalResult::Value(cond) => Terminator::CondJump {
                        cond,
                        if_true: if_true_id,
                        if_true_args: Vec::new(),
                        if_false: if_false_id,
                        if_false_args: Vec::new(),
                    },
                });

                self.current_block_id = if_true_id;
                let if_true_eval = self.eval_block_expr(&if_expr.if_true, expect_type)?;
                let if_true_ty = if_true_eval.ty();
                self.finalize_block(match if_true_eval {
                    EvalResult::Diverges(_) => Terminator::Unreachable,
                    EvalResult::Value(value) => Terminator::Jump {
                        to: continuation_id,
                        args: vec![value],
                    },
                });
                self.current_block_id = if_false_id;
                let if_false_eval = self.eval_block_expr(if_false_expr, expect_type)?;
                let if_false_ty = if_false_eval.ty();
                self.finalize_block(match if_false_eval {
                    EvalResult::Diverges(_) => Terminator::Unreachable,
                    EvalResult::Value(value) => Terminator::Jump {
                        to: continuation_id,
                        args: vec![value],
                    },
                });
                self.current_block_id = continuation_id;

                let Some(result_ty) = if_true_ty.comine_ignoring_never(if_false_ty) else {
                    return Err(Error::new(format!(
                        "if expression branches evaluate to different types: {if_true_ty:?} and {if_false_ty:?}"
                    ))
                    .with_span(if_expr.if_keyword_span));
                };

                let value = DefinitionId::new(result_ty);
                self.current_block_args.push(value);
                Ok(EvalResult::Value(Value::Definition(value)))
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
                    Ok(EvalResult::Diverges(Type::Never))
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
                self.finalize_block(match cond_eval {
                    EvalResult::Diverges(_) => Terminator::Unreachable,
                    EvalResult::Value(cond) => Terminator::CondJump {
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
            ast::ExprWithBlock::StructInitializer(e) => {
                let (ty, sid) = match &e.struct_name {
                    Some(name) => {
                        let ty =
                            self.type_namespace.get(&name.value).copied().ok_or_else(|| {
                                Error::new(format!("unknown type {:?}", name.value)).with_span(name.span)
                            })?;
                        let sid = match ty {
                            Type::Struct(sid) => sid,
                            other => {
                                return Err(
                                    Error::new(format!("{} is not a struct type, but {other:?}", name.value))
                                        .with_span(name.span),
                                );
                            }
                        };
                        if let Some(expect_type) = expect_type
                            && expect_type != ty
                        {
                            return Err(Error::expr_type_missmatch(expect_type, ty, expr.span()));
                        }
                        (ty, sid)
                    }
                    None => {
                        let expect_type =
                            expect_type.ok_or_else(|| Error::new("type annotations needed").with_span(expr.span()))?;
                        let sid = match expect_type {
                            Type::Struct(sid) => sid,
                            other => {
                                return Err(Error::new(format!(
                                    "expected expr of type {other:?}, got struct initializer"
                                ))
                                .with_span(expr.span()));
                            }
                        };
                        (expect_type, sid)
                    }
                };
                let struct_def = self.typesystem.get_struct(sid);
                if let Some(missing_field) = struct_def
                    .fields
                    .iter()
                    .find(|f| !e.fields.iter().any(|ef| ef.name.value == f.name.value))
                {
                    return Err(Error::new(format!("missing field: {}", missing_field.name.value))
                        .with_span(e.opening_brace_span.join(e.closing_brace_span)));
                }
                let place = self.alloca(ty);
                for ef in &e.fields {
                    let (offset, f_ty) = self.typesystem.get_struct_field(sid, &ef.name)?;
                    match self.eval_expr(&ef.value, Some(f_ty))? {
                        EvalResult::Diverges(_) => (),
                        EvalResult::Value(value) => {
                            let ptr = self
                                .cursor()
                                .offset_ptr(Value::Definition(place), offset.try_into().unwrap());
                            self.cursor().store(ptr, value);
                        }
                    }
                }
                Ok(EvalResult::Value(Value::Definition(
                    self.cursor().load(Value::Definition(place), ty),
                )))
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

    /// Generate a `Cmp` instruction
    fn cmp(&mut self, op: CmpOp, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::Cmp { op, lhs, rhs },
        });
        definition_id
    }

    /// Generate an `Arithmetic` instruction
    fn arithmetic(&mut self, op: ArithmeticOp, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(lhs.ty());
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::Arithmetic { op, lhs, rhs },
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
    fn offset_ptr(&mut self, ptr: Value, offset: i64) -> Value {
        if offset == 0 {
            ptr
        } else {
            let definition_id = DefinitionId::new(Type::OpaquePointer);
            self.buf.push(Instruction {
                definition_id,
                kind: InstructionKind::OffsetPtr { ptr, offset },
            });
            Value::Definition(definition_id)
        }
    }
}
