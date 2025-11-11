use std::collections::HashMap;

use super::*;
use crate::ast;

/// Construct an IR of a function from its AST
pub fn build_function(
    decl: &FunctionDecl,
    body: &ast::BlockExpr,
    function_decls: &HashMap<String, FunctionDecl>,
) -> Result<Function, Error> {
    let mut builder = FunctionBuilder::new(decl.return_ty, function_decls);

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
    parent: Option<Box<Self>>,
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
}

/// The result of an expression evaluation
#[derive(Debug)]
struct EvalResult {
    ty: Type,
    value: MaybeValue,
}

impl EvalResult {
    /// Returns `true` if the expression deverges
    fn diverges(&self) -> bool {
        matches!(self.value, MaybeValue::Diverges)
    }
}

/// The value of an exprission, possibly missing due to the expression being diverging
#[derive(Debug)]
enum MaybeValue {
    Diverges,
    Value(Value),
}

impl<'a> FunctionBuilder<'a> {
    /// Create a new builder context
    fn new(return_ty: Type, function_decls: &'a HashMap<String, FunctionDecl>) -> Self {
        Self {
            return_ty,
            function_decls,
            allocas: Vec::new(),
            basic_blocks: HashMap::new(),
            current_block_id: BasicBlockId::new(),
            current_block_args: Vec::new(),
            current_instructions: Vec::new(),
            scope: Scope::default(),
        }
    }

    /// Finalize the current basic block, and start editing a new empty basic block
    fn finalize_block(&mut self, terminator: Terminator) -> BasicBlockId {
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
        let retval = self.current_block_id;
        self.current_block_id = BasicBlockId::new();
        retval
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
            size: ty.size(),
            align: ty.align(),
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
            ast::Expr::WithNoBlock(ast::ExprWithNoBlock::Ident(ident)) => {
                match self.scope.lookup_variable(&ident.value) {
                    Some((alloca, ty)) => Ok(EvalResult {
                        ty,
                        value: MaybeValue::Value(Value::Definition(alloca)),
                    }),
                    None => Err(Error::new(format!("variable {:?} not found", ident.value))
                        .with_span(ident.span)),
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
        let mut diverges = false;
        let mut value = None;

        for (i, stmt) in expr.statements.iter().enumerate() {
            match stmt {
                ast::Statement::Empty => (),
                ast::Statement::Let(let_statement) => match let_statement {
                    ast::LetStatement::WithValue { name, ty, value } => {
                        let ty = ty.as_ref().map(Type::from_ast).transpose()?;
                        let value_eval = self.eval_expr(value, ty)?;
                        let alloca = self.alloca(value_eval.ty);
                        self.scope
                            .variables
                            .insert(name.value.clone(), (alloca, value_eval.ty));
                        match value_eval.value {
                            MaybeValue::Diverges => diverges = true,
                            MaybeValue::Value(value) => {
                                self.cursor().store(Value::Definition(alloca), value);
                            }
                        }
                    }
                    ast::LetStatement::WithoutValue { name, ty } => {
                        let ty = Type::from_ast(ty)?;
                        let alloca = self.alloca(ty);
                        self.scope
                            .variables
                            .insert(name.value.clone(), (alloca, ty));
                    }
                },
                ast::Statement::ExprWithNoBlock(expr_with_no_block) => {
                    diverges |= self
                        .eval_expr_with_no_block(expr_with_no_block, None)?
                        .diverges();
                }
                ast::Statement::ExprWithBlock(expr_with_block) => {
                    if i + 1 == expr.statements.len() && expr.final_expr.is_none() {
                        value = Some(self.eval_expr_with_block(expr_with_block, expect_type)?);
                    } else {
                        diverges |= self
                            .eval_expr_with_block(expr_with_block, Some(Type::Void))?
                            .diverges();
                    }
                }
            }
        }

        if let Some(final_expr) = &expr.final_expr {
            value = Some(self.eval_expr_with_no_block(final_expr, expect_type)?);
        }

        self.scope.pop();

        Ok(match value {
            Some(mut value) => {
                if diverges {
                    value.value = MaybeValue::Diverges;
                }
                value
            }
            None => {
                if diverges {
                    EvalResult {
                        ty: expect_type.unwrap_or(Type::Never),
                        value: MaybeValue::Diverges,
                    }
                } else {
                    if let Some(expect_type) = expect_type
                        && expect_type != Type::Void
                    {
                        return Err(Error::new(format!(
                            "expectd expr of type {expect_type:?}, found end-of-block"
                        ))
                        .with_span(expr.closing_brace_span));
                    }
                    EvalResult {
                        ty: Type::Void,
                        value: MaybeValue::Value(Value::Constant(Constant::Void)),
                    }
                }
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
                    MaybeValue::Value(value) => {
                        self.finalize_block(Terminator::Return { value });
                    }
                }
                Ok(EvalResult {
                    ty: Type::Never,
                    value: MaybeValue::Diverges,
                })
            }
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
                            | Type::OpaquePointer => {
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
                if decl.args.len() != function_call_expr.args.len() {
                    return Err(Error::new(format!(
                        "expected {} arguments, found {}",
                        decl.args.len(),
                        function_call_expr.args.len()
                    ))
                    .with_span(function_call_expr.args_span));
                }
                for (arg_i, arg_expr) in function_call_expr.args.iter().enumerate() {
                    let expect_arg_type = decl.args[arg_i].ty;
                    let arg_eval = self.eval_expr(arg_expr, Some(expect_arg_type))?;
                    if !args_diverges {
                        match arg_eval.value {
                            MaybeValue::Diverges => args_diverges = true,
                            MaybeValue::Value(value) => args_values.push(value),
                        }
                    }
                }
                if let Some(expect_type) = expect_type
                    && expect_type != decl.return_ty
                {
                    return Err(Error::expr_type_missmatch(
                        expect_type,
                        decl.return_ty,
                        function_call_expr.expr_span,
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
                    Ok(EvalResult {
                        ty: decl.return_ty,
                        value: MaybeValue::Value(Value::Definition(definition_id)),
                    })
                }
            }
            ast::ExprWithNoBlock::Assignment(assignment_expr) => {
                let place_eval = self.eval_place_expr(&assignment_expr.place)?;
                let value_eval = self.eval_expr(&assignment_expr.value, Some(place_eval.ty))?;
                if let MaybeValue::Value(ptr) = place_eval.value
                    && let MaybeValue::Value(value) = value_eval.value
                {
                    self.cursor().store(ptr, value);
                    Ok(EvalResult {
                        ty: Type::Void,
                        value: MaybeValue::Value(Value::Constant(Constant::Void)),
                    })
                } else {
                    Ok(EvalResult {
                        ty: Type::Void,
                        value: MaybeValue::Diverges,
                    })
                }
            }
            ast::ExprWithNoBlock::Ident(ident) => match self.scope.lookup_variable(&ident.value) {
                Some((alloca, ty)) => {
                    if let Some(expect_type) = expect_type
                        && expect_type != ty
                    {
                        return Err(Error::expr_type_missmatch(expect_type, ty, ident.span));
                    }
                    let definition_id = self.cursor().load(Value::Definition(alloca), ty);
                    Ok(EvalResult {
                        ty,
                        value: MaybeValue::Value(Value::Definition(definition_id)),
                    })
                }
                None => Err(Error::new(format!("variable {:?} not found", ident.value))
                    .with_span(ident.span)),
            },
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
                            MaybeValue::Value(Value::Definition(
                                match lhs_eval.ty.is_signed_int() {
                                    true => self.cursor().cmp_sl(lhs, rhs),
                                    false => self.cursor().cmp_ul(lhs, rhs),
                                },
                            ))
                        } else {
                            MaybeValue::Diverges
                        },
                    })
                }
                ast::BinaryOp::Greater => todo!(),
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
                        | Type::OpaquePointer => {
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
            },
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
                let cond_diverges = match cond.value {
                    MaybeValue::Diverges => {
                        self.finalize_block(Terminator::Unreachable);
                        true
                    }
                    MaybeValue::Value(cond) => {
                        self.finalize_block(Terminator::CondJump {
                            cond,
                            if_true: if_true_id,
                            if_true_args: Vec::new(),
                            if_false: continuation_id,
                            if_false_args: Vec::new(),
                        });
                        false
                    }
                };

                self.current_block_id = if_true_id;
                self.eval_block_expr(&if_expr.if_true, Some(Type::Void))?;
                self.finalize_block(Terminator::Jump {
                    to: continuation_id,
                    args: Vec::new(),
                });
                self.current_block_id = continuation_id;

                Ok(EvalResult {
                    ty: Type::Void,
                    value: match cond_diverges {
                        true => MaybeValue::Diverges,
                        false => MaybeValue::Value(Value::Constant(Constant::Void)),
                    },
                })
            }
            ast::ExprWithBlock::If(if_expr) => {
                let if_false_expr = if_expr.if_false.as_ref().unwrap();

                let continuation_id = BasicBlockId::new();
                let if_true_id = BasicBlockId::new();
                let if_false_id = BasicBlockId::new();

                let cond = self.eval_expr(&if_expr.cond, Some(Type::Bool))?;
                let cond_diverges = match cond.value {
                    MaybeValue::Diverges => {
                        self.finalize_block(Terminator::Unreachable);
                        true
                    }
                    MaybeValue::Value(cond) => {
                        self.finalize_block(Terminator::CondJump {
                            cond,
                            if_true: if_true_id,
                            if_true_args: Vec::new(),
                            if_false: if_false_id,
                            if_false_args: Vec::new(),
                        });
                        false
                    }
                };

                self.current_block_id = if_true_id;
                let if_true_eval = self.eval_block_expr(&if_expr.if_true, expect_type)?;
                let if_true_diverges = match if_true_eval.value {
                    MaybeValue::Diverges => {
                        self.finalize_block(Terminator::Unreachable);
                        true
                    }
                    MaybeValue::Value(value) => {
                        self.finalize_block(Terminator::Jump {
                            to: continuation_id,
                            args: vec![value],
                        });
                        false
                    }
                };
                self.current_block_id = if_false_id;
                let if_false_eval = self.eval_block_expr(if_false_expr, expect_type)?;
                let if_false_diverges = match if_false_eval.value {
                    MaybeValue::Diverges => {
                        self.finalize_block(Terminator::Unreachable);
                        true
                    }
                    MaybeValue::Value(value) => {
                        self.finalize_block(Terminator::Jump {
                            to: continuation_id,
                            args: vec![value],
                        });
                        false
                    }
                };
                self.current_block_id = continuation_id;

                let Some(result_ty) = if_true_eval.ty.comine_ignoring_never(if_false_eval.ty)
                else {
                    return Err(Error::new(format!(
                        "if expression branches evaluate to different types: {:?} and {:?}",
                        if_true_eval.ty, if_false_eval.ty
                    ))
                    .with_span(if_expr.if_keyword_span));
                };

                if cond_diverges || (if_true_diverges && if_false_diverges) {
                    Ok(EvalResult {
                        ty: result_ty,
                        value: MaybeValue::Diverges,
                    })
                } else {
                    let value = DefinitionId::new(result_ty);
                    self.current_block_args.push(value);
                    Ok(EvalResult {
                        ty: result_ty,
                        value: MaybeValue::Value(Value::Definition(value)),
                    })
                }
            }
            ast::ExprWithBlock::Loop(loop_expr) => {
                let body_id = BasicBlockId::new();
                self.finalize_block(Terminator::Jump {
                    to: body_id,
                    args: Vec::new(),
                });
                self.current_block_id = body_id;

                let _body_eval = self.eval_block_expr(&loop_expr.body, Some(Type::Void))?;
                self.finalize_block(Terminator::Jump {
                    to: body_id,
                    args: Vec::new(),
                });
                // TODO: diverges if body_eval diverges, even if breaks are inside.

                Ok(EvalResult {
                    ty: Type::Never,
                    value: MaybeValue::Diverges,
                })
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

    /// Generate a `CmpSL` instruction
    fn cmp_sl(&mut self, lhs: Value, rhs: Value) -> DefinitionId {
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::CmpSL { lhs, rhs },
        });
        definition_id
    }

    /// Generate a `CmpUL` instruction
    fn cmp_ul(&mut self, lhs: Value, rhs: Value) -> DefinitionId {
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id,
            kind: InstructionKind::CmpUL { lhs, rhs },
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
}
