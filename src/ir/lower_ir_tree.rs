use std::collections::HashMap;

use super::*;
use crate::ir_tree;

/// Lower IR_TREE function to IR function.
pub fn lower_function(function: &ir_tree::Function, module: &ir_tree::Module) -> Result<Function, Error> {
    let mut ir_function = Function {
        mangled_name: function.name.value.clone(),
        args: function
            .args
            .iter()
            .map(|(_name, ty)| lower_type(module, *ty))
            .collect(),
        is_variadic: function.is_variadic,
        never_returs: function.return_ty == ir_tree::Type::Never,
        return_ty: lower_type(module, function.return_ty),
        body: None,
    };

    if let Some(body) = &function.body {
        if function.is_variadic {
            return Err(Error::new("defining variadic functions is not supported").with_span(function.name.span));
        }
        ir_function.body = Some(lower_function_body(module, &ir_function, function, body)?);
    }

    Ok(ir_function)
}

/// Lower IR_TREE type to IR type
fn lower_type(module: &ir_tree::Module, ty: ir_tree::Type) -> Type {
    match ty {
        ir_tree::Type::Never | ir_tree::Type::Unit => Type::Unit,
        ir_tree::Type::Bool => Type::Bool,
        ir_tree::Type::Ptr { .. } => Type::Ptr,
        ir_tree::Type::Int(int_type) => match int_type {
            ir_tree::IntType::I8 | ir_tree::IntType::U8 => Type::I8,
            ir_tree::IntType::I32 | ir_tree::IntType::U32 => Type::I32,
            ir_tree::IntType::I64 | ir_tree::IntType::U64 => Type::I64,
        },
        ir_tree::Type::Array { element, length } => {
            let element = lower_type(module, module.typesystem.get_type(element));
            Type::Array(Box::new(element), length)
        }
        ir_tree::Type::Struct(struct_id) => {
            let fields = module
                .typesystem
                .get_struct(struct_id)
                .fields
                .iter()
                .map(|field| lower_type(module, field.ty))
                .collect();
            Type::Struct(fields)
        }
    }
}

fn lower_function_body(
    module: &ir_tree::Module,
    function: &Function,
    ir_tree_function: &ir_tree::Function,
    body: &ir_tree::Expr,
) -> Result<FunctionBody, Error> {
    let mut builder = BodyLoweringCtx::new(module, function, ir_tree_function);

    let entry = builder.current_block_id;
    let body_eval = builder.eval_expr(body)?;
    builder.finalize_block(match body_eval {
        EvalResult::Never => Terminator::Unreachable,
        EvalResult::Value(value) => Terminator::Return(value),
    });

    Ok(FunctionBody {
        allocas: builder.allocas,
        entry,
        basic_blokcs: builder.basic_blocks,
    })
}

/// IR_TREE -> IR function body lowering context
struct BodyLoweringCtx<'a> {
    module: &'a ir_tree::Module,
    allocas: Vec<Alloca>,
    argument_map: HashMap<String, DefinitionId>,
    variable_map: HashMap<ir_tree::VariableId, DefinitionId>,
    basic_blocks: HashMap<BasicBlockId, BasicBlock>,
    current_block_id: BasicBlockId,
    current_block_args: Vec<DefinitionId>,
    current_instructions: Vec<Instruction>,
    break_target_map: HashMap<ir_tree::LoopId, BasicBlockId>,
}

enum EvalResult<T = Value> {
    Never,
    Value(T),
}

impl From<DefinitionId> for EvalResult<Value> {
    fn from(value: DefinitionId) -> Self {
        Self::Value(Value::Definition(value))
    }
}

impl<'a> BodyLoweringCtx<'a> {
    /// Create a new lowering context
    fn new(module: &'a ir_tree::Module, function: &'a Function, ir_tree_function: &'a ir_tree::Function) -> Self {
        let mut argument_map = HashMap::new();
        let mut entry_block_args = Vec::new();
        for (arg_i, arg_ty) in function.args.iter().enumerate() {
            let arg_name = ir_tree_function.args[arg_i].0.clone();
            let def_id = DefinitionId::new(arg_ty.clone());
            argument_map.insert(arg_name, def_id.clone());
            entry_block_args.push(def_id);
        }

        Self {
            module,
            allocas: Vec::new(),
            argument_map,
            variable_map: HashMap::new(),
            basic_blocks: HashMap::new(),
            current_block_id: BasicBlockId::new(),
            current_block_args: entry_block_args,
            current_instructions: Vec::new(),
            break_target_map: HashMap::new(),
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
    fn alloca(&mut self, layout: Layout) -> DefinitionId {
        let alloca = DefinitionId::new(Type::Ptr);
        self.allocas.push(Alloca {
            definition_id: alloca.clone(),
            layout,
        });
        alloca
    }

    fn eval_expr(&mut self, expr: &ir_tree::Expr) -> Result<EvalResult, Error> {
        match expr {
            ir_tree::Expr::R(rexpr) => self.eval_rexpr(rexpr),
            ir_tree::Expr::L(lexpr) => self.eval_lexpr(lexpr),
        }
    }

    /// Evaluate an r-value expression
    fn eval_rexpr(&mut self, expr: &ir_tree::RExpr) -> Result<EvalResult, Error> {
        let ty = lower_type(self.module, expr.ty);
        Ok(match &expr.kind {
            ir_tree::RExprKind::Undefined => EvalResult::Value(Value::Undefined(ty)),
            ir_tree::RExprKind::ConstUnit => EvalResult::Value(Value::Zst),
            ir_tree::RExprKind::ConstNumber(num) => EvalResult::Value(Value::Number { data: *num, ty }),
            ir_tree::RExprKind::ConstString(str) => EvalResult::Value(Value::String(str.clone())),
            ir_tree::RExprKind::ConstBool(bool) => EvalResult::Value(Value::Bool(*bool)),

            ir_tree::RExprKind::Field(..)
            | ir_tree::RExprKind::ArrayElement(..)
            | ir_tree::RExprKind::StructInitializer(..)
            | ir_tree::RExprKind::ArrayInitializer(..) => match self.eval_rexpr_as_tmp_ptr(expr)? {
                EvalResult::Never => EvalResult::Never,
                EvalResult::Value(ptr) => EvalResult::Value(Value::Definition(self.cursor().load(ptr, ty))),
            },

            ir_tree::RExprKind::Store(place, value) => {
                let place_ptr = match self.eval_lexpr_as_ptr(place)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let value = match self.eval_expr(value)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                self.cursor().store(place_ptr, value);
                EvalResult::Value(Value::Zst)
            }
            ir_tree::RExprKind::GetPointer(lexpr) => self.eval_lexpr_as_ptr(lexpr)?,

            ir_tree::RExprKind::Argument(arg_name) => {
                let arg_def_id = self.argument_map[arg_name].clone();
                EvalResult::Value(Value::Definition(arg_def_id))
            }
            ir_tree::RExprKind::Block(block_expr) => {
                for (var_id, var_ty) in &block_expr.variables {
                    let var_ty = lower_type(self.module, *var_ty);
                    let alloca = self.alloca(var_ty.layout(self.module));
                    self.variable_map.insert(*var_id, alloca);
                }
                for (expr_i, expr) in block_expr.exprs.iter().enumerate() {
                    match self.eval_expr(expr)? {
                        EvalResult::Never => return Ok(EvalResult::Never),
                        result if expr_i + 1 == block_expr.exprs.len() => return Ok(result),
                        _ => (),
                    }
                }
                EvalResult::Value(Value::Zst)
            }
            ir_tree::RExprKind::Return(return_expr) => match self.eval_expr(return_expr)? {
                EvalResult::Never => EvalResult::Never,
                EvalResult::Value(value) => {
                    self.finalize_block(Terminator::Return(value));
                    EvalResult::Never
                }
            },
            ir_tree::RExprKind::Break(break_from, value) => match self.eval_expr(value)? {
                EvalResult::Never => EvalResult::Never,
                EvalResult::Value(value) => {
                    let to = self.break_target_map[break_from];
                    self.finalize_block(Terminator::Jump { to, args: vec![value] });
                    EvalResult::Never
                }
            },
            ir_tree::RExprKind::Arithmetic(op, lhs, rhs) => {
                let signed = lhs.ty().is_signed_int();
                let lhs = match self.eval_expr(lhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let rhs = match self.eval_expr(rhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(Value::Definition(self.cursor().arithmetic(*op, signed, lhs, rhs)))
            }
            ir_tree::RExprKind::Cmp(op, lhs, rhs) => {
                let signed = lhs.ty().is_signed_int();
                let lhs = match self.eval_expr(lhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let rhs = match self.eval_expr(rhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(Value::Definition(self.cursor().cmp(*op, signed, lhs, rhs)))
            }
            ir_tree::RExprKind::If {
                cond,
                if_true,
                if_false,
            } => {
                let continuation_id = BasicBlockId::new();
                let if_true_id = BasicBlockId::new();
                let if_false_id = BasicBlockId::new();

                let cond = match self.eval_expr(cond)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };

                self.finalize_block(Terminator::CondJump {
                    cond,
                    if_true: if_true_id,
                    if_true_args: Vec::new(),
                    if_false: if_false_id,
                    if_false_args: Vec::new(),
                });

                self.current_block_id = if_true_id;
                match self.eval_expr(if_true)? {
                    EvalResult::Never => (),
                    EvalResult::Value(val) => self.finalize_block(Terminator::Jump {
                        to: continuation_id,
                        args: vec![val],
                    }),
                }

                self.current_block_id = if_false_id;
                match self.eval_expr(if_false)? {
                    EvalResult::Never => (),
                    EvalResult::Value(val) => self.finalize_block(Terminator::Jump {
                        to: continuation_id,
                        args: vec![val],
                    }),
                }

                let value = DefinitionId::new(ty);
                self.current_block_id = continuation_id;
                self.current_block_args.push(value.clone());
                EvalResult::Value(Value::Definition(value))
            }
            ir_tree::RExprKind::Loop(loop_id, body) => {
                let body_id = BasicBlockId::new();
                let continuation_id = BasicBlockId::new();

                self.break_target_map.insert(*loop_id, continuation_id);

                self.finalize_block(Terminator::Jump {
                    to: body_id,
                    args: Vec::new(),
                });

                self.current_block_id = body_id;
                self.eval_expr(body)?;
                self.finalize_block(Terminator::Jump {
                    to: body_id,
                    args: Vec::new(),
                });

                let value = DefinitionId::new(ty);
                self.current_block_id = continuation_id;
                self.current_block_args.push(value.clone());
                if expr.ty == ir_tree::Type::Never {
                    EvalResult::Never
                } else {
                    EvalResult::Value(Value::Definition(value))
                }
            }
            ir_tree::RExprKind::FunctionCall(function_id, args) => {
                let mut arg_vals = Vec::new();
                for arg_expr in args {
                    match self.eval_expr(arg_expr)? {
                        EvalResult::Never => return Ok(EvalResult::Never),
                        EvalResult::Value(val) => arg_vals.push(val),
                    }
                }
                let name = self.module.functions[function_id].name.value.clone();
                let val_def_id = self.cursor().function_call(name, arg_vals, ty);
                if expr.ty == ir_tree::Type::Never {
                    self.finalize_block(Terminator::Unreachable);
                    EvalResult::Never
                } else {
                    EvalResult::Value(Value::Definition(val_def_id))
                }
            }
            ir_tree::RExprKind::Cast(value) => {
                let from_ty = value.ty();
                let to_ty = expr.ty;
                let value = match self.eval_expr(value)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                if value.ty() == ty {
                    EvalResult::Value(value)
                } else if from_ty.is_int() && to_ty.is_int() {
                    let ir_tree::Type::Int(from_ty) = from_ty else {
                        unreachable!()
                    };
                    let ir_tree::Type::Int(to_ty) = to_ty else {
                        unreachable!()
                    };
                    EvalResult::Value(Value::Definition(if from_ty.bytes() > to_ty.bytes() {
                        self.cursor().truncate(value, ty)
                    } else if !from_ty.is_signed() && !to_ty.is_signed() {
                        self.cursor().zext(value, ty)
                    } else {
                        self.cursor().sext(value, ty)
                    }))
                } else {
                    unimplemented!("cast from {from_ty:?} to {to_ty:?} is not handled")
                }
            }
            ir_tree::RExprKind::Not(value) => {
                let value = match self.eval_expr(value)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(Value::Definition(self.cursor().not(value)))
            }
        })
    }

    fn eval_rexpr_as_tmp_ptr(&mut self, expr: &ir_tree::RExpr) -> Result<EvalResult, Error> {
        let ty = lower_type(self.module, expr.ty);
        Ok(match &expr.kind {
            ir_tree::RExprKind::Field(lhs, field) => {
                let field_offset = lhs.ty.get_field_offset(field, &self.module.typesystem).unwrap();
                let lhs_ptr = match self.eval_rexpr_as_tmp_ptr(lhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(self.cursor().offset_ptr(lhs_ptr, Value::new_i64(field_offset as i64)))
            }
            ir_tree::RExprKind::ArrayElement(array, index) => {
                let array_ptr = match self.eval_rexpr_as_tmp_ptr(array)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let index = match self.eval_expr(index)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let element_layout = ty.layout(self.module);
                let ptr_offset = self.cursor().arithmetic(
                    ArithmeticOp::Mul,
                    false,
                    index,
                    Value::new_i64(element_layout.size as i64),
                );
                EvalResult::Value(self.cursor().offset_ptr(array_ptr, Value::Definition(ptr_offset)))
            }
            ir_tree::RExprKind::ArrayInitializer(exprs) => {
                let mut elements = Vec::new();
                for expr in exprs {
                    match self.eval_expr(expr)? {
                        EvalResult::Never => return Ok(EvalResult::Never),
                        EvalResult::Value(element) => elements.push(element),
                    }
                }

                let alloca = Value::Definition(self.alloca(ty.layout(self.module)));
                let element_layout = ty.array_element_type().unwrap().layout(self.module);
                for (i, element) in elements.into_iter().enumerate() {
                    let ptr = self
                        .cursor()
                        .offset_ptr(alloca.clone(), Value::new_i64(i as i64 * element_layout.size as i64));
                    self.cursor().store(ptr, element);
                }

                EvalResult::Value(alloca)
            }
            ir_tree::RExprKind::StructInitializer(fields) => {
                let mut exprs = Vec::new();
                for (field_name, field_expr) in fields {
                    match self.eval_expr(field_expr)? {
                        EvalResult::Never => return Ok(EvalResult::Never),
                        EvalResult::Value(val) => {
                            let offset = expr.ty.get_field_offset(field_name, &self.module.typesystem).unwrap();
                            exprs.push((offset, val));
                        }
                    }
                }

                let alloca = Value::Definition(self.alloca(ty.layout(self.module)));
                for (offset, value) in exprs {
                    let ptr = self.cursor().offset_ptr(
                        alloca.clone(),
                        Value::Number {
                            data: offset as i64,
                            ty: Type::I64,
                        },
                    );
                    self.cursor().store(ptr, value);
                }

                EvalResult::Value(alloca)
            }
            _ => {
                let value = match self.eval_rexpr(expr)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let alloca = self.alloca(ty.layout(self.module));
                self.cursor().store(Value::Definition(alloca.clone()), value);
                EvalResult::Value(Value::Definition(alloca))
            }
        })
    }

    fn eval_lexpr(&mut self, expr: &ir_tree::LExpr) -> Result<EvalResult, Error> {
        let ty = lower_type(self.module, expr.ty);
        Ok(match self.eval_lexpr_as_ptr(expr)? {
            EvalResult::Never => EvalResult::Never,
            EvalResult::Value(ptr) => EvalResult::Value(Value::Definition(self.cursor().load(ptr, ty))),
        })
    }

    fn eval_lexpr_as_ptr(&mut self, expr: &ir_tree::LExpr) -> Result<EvalResult, Error> {
        Ok(match &expr.kind {
            ir_tree::LExprKind::Dereference(ptr) => self.eval_expr(ptr)?,
            ir_tree::LExprKind::Variable(var_id) => {
                EvalResult::Value(Value::Definition(self.variable_map[var_id].clone()))
            }
            ir_tree::LExprKind::Field(place, field_name) => {
                let field_offset = place.ty.get_field_offset(field_name, &self.module.typesystem).unwrap();
                let place_ptr = match self.eval_lexpr_as_ptr(place)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(self.cursor().offset_ptr(place_ptr, Value::new_i64(field_offset as i64)))
            }
            ir_tree::LExprKind::ArrayElement(array, index) => {
                let array_ptr = match self.eval_lexpr_as_ptr(array)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let index = match self.eval_expr(index)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let element_layout = array
                    .ty
                    .array_element_type(&self.module.typesystem)
                    .unwrap()
                    .layout(&self.module.typesystem);
                let ptr_offset = self.cursor().arithmetic(
                    ArithmeticOp::Mul,
                    false,
                    index,
                    Value::new_i64(element_layout.size as i64),
                );
                EvalResult::Value(self.cursor().offset_ptr(array_ptr, Value::Definition(ptr_offset)))
            }
        })
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
            definition_id: definition_id.clone(),
            kind: InstructionKind::Load { ptr },
        });
        definition_id
    }

    /// Generate a `Store` instruction
    fn store(&mut self, ptr: Value, value: Value) {
        self.buf.push(Instruction {
            definition_id: DefinitionId::new(Type::Unit),
            kind: InstructionKind::Store { ptr, value },
        });
    }

    /// Generate a `FunctionCall` instruction
    fn function_call(&mut self, name: String, args: Vec<Value>, ty: Type) -> DefinitionId {
        let definition_id = DefinitionId::new(ty);
        self.buf.push(Instruction {
            definition_id: definition_id.clone(),
            kind: InstructionKind::FunctionCall { name, args },
        });
        definition_id
    }

    /// Generate a `Cmp` instruction
    fn cmp(&mut self, op: CmpOp, signed: bool, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id: definition_id.clone(),
            kind: InstructionKind::Cmp { op, signed, lhs, rhs },
        });
        definition_id
    }

    /// Generate an `Arithmetic` instruction
    fn arithmetic(&mut self, op: ArithmeticOp, signed: bool, lhs: Value, rhs: Value) -> DefinitionId {
        assert_eq!(lhs.ty(), rhs.ty());
        let definition_id = DefinitionId::new(lhs.ty());
        self.buf.push(Instruction {
            definition_id: definition_id.clone(),
            kind: InstructionKind::Arithmetic { op, signed, lhs, rhs },
        });
        definition_id
    }

    /// Generate a `Not` instruction
    fn not(&mut self, value: Value) -> DefinitionId {
        assert_eq!(value.ty(), Type::Bool);
        let definition_id = DefinitionId::new(Type::Bool);
        self.buf.push(Instruction {
            definition_id: definition_id.clone(),
            kind: InstructionKind::Not { value },
        });
        definition_id
    }

    /// Generate a `OffsetPtr` instruction
    fn offset_ptr(&mut self, ptr: Value, offset: Value) -> Value {
        if matches!(offset, Value::Number { data: 0, ty: _ }) {
            ptr
        } else {
            let definition_id = DefinitionId::new(Type::Ptr);
            self.buf.push(Instruction {
                definition_id: definition_id.clone(),
                kind: InstructionKind::OffsetPtr { ptr, offset },
            });
            Value::Definition(definition_id)
        }
    }

    /// Generate a `Zext` instruction
    fn zext(&mut self, int: Value, target_int_ty: Type) -> DefinitionId {
        let definition_id = DefinitionId::new(target_int_ty);
        self.buf.push(Instruction {
            definition_id: definition_id.clone(),
            kind: InstructionKind::Zext { int },
        });
        definition_id
    }

    /// Generate a `Sext` instruction
    fn sext(&mut self, int: Value, target_int_ty: Type) -> DefinitionId {
        let definition_id = DefinitionId::new(target_int_ty);
        self.buf.push(Instruction {
            definition_id: definition_id.clone(),
            kind: InstructionKind::Sext { int },
        });
        definition_id
    }

    /// Generate a `Truncate` instruction
    fn truncate(&mut self, int: Value, target_int_ty: Type) -> DefinitionId {
        let definition_id = DefinitionId::new(target_int_ty);
        self.buf.push(Instruction {
            definition_id: definition_id.clone(),
            kind: InstructionKind::Truncate { int },
        });
        definition_id
    }
}
