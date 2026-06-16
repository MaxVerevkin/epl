use std::collections::HashMap;

use super::*;
use crate::ir_tree;

/// Lower IR_TREE function to IR function.
pub fn lower_function<'ctx>(
    function: &ir_tree::Function<'ctx>,
    module: &ir_tree::Module<'ctx>,
) -> Result<Function, Error> {
    let mut ir_function = Function {
        mangled_name: function.name.value.clone(),
        args: function
            .args
            .iter()
            .map(|(_name, ty)| lower_type(module, *ty))
            .collect(),
        is_variadic: function.is_variadic,
        never_returns: function.return_ty.is_never(),
        return_ty: lower_type(module, function.return_ty),
        body: None,
    };

    if let Some(body) = &function.body {
        if function.is_variadic {
            return Err(Error::new("defining variadic functions is not supported").with_span(function.name.span));
        }
        ir_function.body = Some(lower_function_body(module, &ir_function, body)?);
    }

    Ok(ir_function)
}

/// Lower IR_TREE type to IR type
fn lower_type<'ctx>(module: &ir_tree::Module<'ctx>, ty: ir_tree::Type<'ctx>) -> Type {
    match ty.info() {
        ir_tree::TypeInfo::Never | ir_tree::TypeInfo::Unit => Type::Unit,
        ir_tree::TypeInfo::Bool => Type::Bool,
        ir_tree::TypeInfo::Ptr { .. } => Type::Ptr,
        ir_tree::TypeInfo::Int(int_type) => match int_type.bytes(module.ctx) {
            1 => Type::I8,
            4 => Type::I32,
            8 => Type::I64,
            other => panic!("unsupported integer size: {other} bytes"),
        },
        ir_tree::TypeInfo::Array { element_ty, length } => {
            let element = lower_type(module, *element_ty);
            Type::Array(Box::new(element), *length)
        }
        ir_tree::TypeInfo::Struct {
            struct_,
            type_arguments,
        } => {
            let fields = struct_
                .info()
                .fields
                .iter()
                .map(|f| lower_type(module, module.ctx.type_of_struct_field(f.1, *type_arguments)))
                .collect::<Vec<_>>();
            Type::Struct(fields, ty.layout(module.ctx))
        }
        ir_tree::TypeInfo::TypeParameter { .. } => {
            panic!("generic type parameters are expected to be instantiated before IR lowering");
        }
    }
}

fn lower_function_body<'ctx>(
    module: &ir_tree::Module<'ctx>,
    function: &Function,
    body: &ir_tree::Expr<'ctx>,
) -> Result<FunctionBody, Error> {
    let mut builder = BodyLoweringCtx::new(module, function);

    let entry = builder.current_block_id;
    let body_eval = builder.eval_expr(body)?;
    builder.finalize_block(match body_eval {
        EvalResult::Never => Terminator::Unreachable,
        EvalResult::Value(value) => Terminator::Return(value),
    });

    Ok(FunctionBody {
        allocas: builder.allocas,
        entry,
        basic_blocks: builder.basic_blocks,
    })
}

/// IR_TREE -> IR function body lowering context
struct BodyLoweringCtx<'a, 'ctx> {
    module: &'a ir_tree::Module<'ctx>,
    allocas: Vec<Alloca>,
    arguments: Vec<DefinitionId>,
    variable_map: HashMap<ir_tree::VariableId, DefinitionId>,
    basic_blocks: HashMap<BasicBlockId, BasicBlock>,
    current_block_id: BasicBlockId,
    current_block_args: Vec<DefinitionId>,
    current_instructions: Vec<Instruction>,
    break_target_map: HashMap<ir_tree::LoopId, BasicBlockId>,
    continue_target_map: HashMap<ir_tree::LoopId, BasicBlockId>,
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

impl<'a, 'ctx> BodyLoweringCtx<'a, 'ctx> {
    /// Create a new lowering context
    fn new(module: &'a ir_tree::Module<'ctx>, function: &'a Function) -> Self {
        let mut arguments = Vec::new();
        let mut entry_block_args = Vec::new();
        for arg_ty in &function.args {
            let def_id = DefinitionId::new(arg_ty.clone());
            arguments.push(def_id.clone());
            entry_block_args.push(def_id);
        }

        Self {
            module,
            allocas: Vec::new(),
            arguments,
            variable_map: HashMap::new(),
            basic_blocks: HashMap::new(),
            current_block_id: BasicBlockId::new(),
            current_block_args: entry_block_args,
            current_instructions: Vec::new(),
            break_target_map: HashMap::new(),
            continue_target_map: HashMap::new(),
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

    /// Evaluate an expression
    fn eval_expr(&mut self, expr: &ir_tree::Expr<'ctx>) -> Result<EvalResult, Error> {
        let ty = lower_type(self.module, expr.ty);
        Ok(match &expr.kind {
            ir_tree::ExprKind::Const(value) => EvalResult::Value(self.eval_const(value)),
            ir_tree::ExprKind::ConstString(str) => EvalResult::Value(Value::String(str.clone())),

            ir_tree::ExprKind::Load(place) => match self.eval_place_as_ptr(place)? {
                EvalResult::Never => EvalResult::Never,
                EvalResult::Value(ptr) => EvalResult::Value(Value::Definition(self.cursor().load(ptr, ty))),
            },

            ir_tree::ExprKind::Field(..)
            | ir_tree::ExprKind::ArrayElement(..)
            | ir_tree::ExprKind::StructInitializer(..)
            | ir_tree::ExprKind::ArrayInitializer(..) => match self.eval_expr_as_readonly_ptr(expr)? {
                EvalResult::Never => EvalResult::Never,
                EvalResult::Value(ptr) => EvalResult::Value(Value::Definition(self.cursor().load(ptr, ty))),
            },

            ir_tree::ExprKind::Store(place, value) => {
                let place_ptr = match self.eval_place_as_ptr(place)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                match value.as_ref() {
                    ir_tree::Expr {
                        kind: ir_tree::ExprKind::Const(value),
                        ..
                    } => {
                        self.eval_const_into(place_ptr, value);
                    }
                    ir_tree::Expr {
                        kind: ir_tree::ExprKind::ArrayInitializer(elements),
                        ..
                    } => {
                        let array_element_ty = lower_type(self.module, value.ty.as_array().unwrap().0);
                        match self.eval_array_initializer_into(place_ptr, elements, &array_element_ty)? {
                            EvalResult::Never => return Ok(EvalResult::Never),
                            EvalResult::Value(_) => (),
                        }
                    }
                    ir_tree::Expr {
                        kind: ir_tree::ExprKind::StructInitializer(fields),
                        ..
                    } => match self.eval_struct_initializer_into(place_ptr, fields, &value.ty)? {
                        EvalResult::Never => return Ok(EvalResult::Never),
                        EvalResult::Value(_) => (),
                    },
                    _ => {
                        let value = match self.eval_expr(value)? {
                            EvalResult::Never => return Ok(EvalResult::Never),
                            EvalResult::Value(val) => val,
                        };
                        self.cursor().store(place_ptr, value);
                    }
                }
                EvalResult::Value(Value::Zst)
            }
            ir_tree::ExprKind::GetPointer(lexpr) => self.eval_place_as_ptr(lexpr)?,

            ir_tree::ExprKind::Argument(arg_index) => {
                let arg_def_id = self.arguments[*arg_index].clone();
                EvalResult::Value(Value::Definition(arg_def_id))
            }
            ir_tree::ExprKind::Block(block_expr) => {
                for decl in &block_expr.variables {
                    let var_ty = lower_type(self.module, decl.ty);
                    let alloca = self.alloca(var_ty.layout(self.module.ctx));
                    self.variable_map.insert(decl.id, alloca);
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
            ir_tree::ExprKind::Return(return_expr) => match self.eval_expr(return_expr)? {
                EvalResult::Never => EvalResult::Never,
                EvalResult::Value(value) => {
                    self.finalize_block(Terminator::Return(value));
                    EvalResult::Never
                }
            },
            ir_tree::ExprKind::Break(loop_id, value) => match self.eval_expr(value)? {
                EvalResult::Never => EvalResult::Never,
                EvalResult::Value(value) => {
                    let to = self.break_target_map[loop_id];
                    self.finalize_block(Terminator::Jump { to, args: vec![value] });
                    EvalResult::Never
                }
            },
            ir_tree::ExprKind::Continue(loop_id) => {
                let to = self.continue_target_map[loop_id];
                self.finalize_block(Terminator::Jump { to, args: Vec::new() });
                EvalResult::Never
            }
            ir_tree::ExprKind::Arithmetic(op, lhs, rhs) => {
                let signed = lhs.ty.is_signed_int();
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
            ir_tree::ExprKind::InPlaceArithmetic(op, lhs, rhs) => {
                let signed = lhs.ty.is_signed_int();
                let operands_ty = lower_type(self.module, lhs.ty);
                let lhs_ptr = match self.eval_place_as_ptr(lhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let rhs = match self.eval_expr(rhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let lhs_before = self.cursor().load(lhs_ptr.clone(), operands_ty);
                let result = self
                    .cursor()
                    .arithmetic(*op, signed, Value::Definition(lhs_before), rhs);
                self.cursor().store(lhs_ptr, Value::Definition(result));
                EvalResult::Value(Value::Zst)
            }
            ir_tree::ExprKind::Cmp(op, lhs, rhs) => {
                let signed = lhs.ty.is_signed_int();
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
            ir_tree::ExprKind::If {
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
            ir_tree::ExprKind::Loop(loop_id, body) => {
                let body_id = BasicBlockId::new();
                let continuation_id = BasicBlockId::new();

                self.break_target_map.insert(*loop_id, continuation_id);
                self.continue_target_map.insert(*loop_id, body_id);

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
                if expr.ty.is_never() {
                    EvalResult::Never
                } else {
                    EvalResult::Value(Value::Definition(value))
                }
            }
            ir_tree::ExprKind::FunctionCall(function_id, args) => {
                let mut arg_vals = Vec::new();
                for arg_expr in args {
                    match self.eval_expr(arg_expr)? {
                        EvalResult::Never => return Ok(EvalResult::Never),
                        EvalResult::Value(val) => arg_vals.push(val),
                    }
                }
                let name = self.module.functions[function_id].name.value.clone();
                let val_def_id = self.cursor().function_call(name, arg_vals, ty);
                if expr.ty.is_never() {
                    self.finalize_block(Terminator::Unreachable);
                    EvalResult::Never
                } else {
                    EvalResult::Value(Value::Definition(val_def_id))
                }
            }
            ir_tree::ExprKind::Cast(value) => {
                let from_ty = value.ty;
                let to_ty = expr.ty;
                let value = match self.eval_expr(value)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(if value.ty() == ty {
                    value
                } else if let Some(from_ty) = from_ty.as_int()
                    && let Some(to_ty) = to_ty.as_int()
                {
                    Value::Definition(if from_ty.bytes(self.module.ctx) > to_ty.bytes(self.module.ctx) {
                        self.cursor().truncate(value, ty)
                    } else if from_ty.is_signed() {
                        self.cursor().sext(value, ty)
                    } else {
                        self.cursor().zext(value, ty)
                    })
                } else {
                    unimplemented!("cast from {from_ty:?} to {to_ty:?} is not handled")
                })
            }
            ir_tree::ExprKind::Not(value) => {
                let value = match self.eval_expr(value)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(Value::Definition(self.cursor().not(value)))
            }
            ir_tree::ExprKind::Comptime(_) => {
                panic!("comptime exprs must have been evaluated before IR_TREE -> IR lowering");
            }
        })
    }

    fn eval_expr_as_readonly_ptr(&mut self, expr: &ir_tree::Expr<'ctx>) -> Result<EvalResult, Error> {
        let ty = lower_type(self.module, expr.ty);
        Ok(match &expr.kind {
            ir_tree::ExprKind::Field(lhs, field_id) => {
                let (_struct_, type_arguments) = lhs.ty.as_struct().unwrap();
                let field_offset = self.module.ctx.offset_of_struct_field(*field_id, type_arguments);
                let lhs_ptr = match self.eval_expr_as_readonly_ptr(lhs)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(self.cursor().offset_ptr(lhs_ptr, Value::new_i64(field_offset as i64)))
            }
            ir_tree::ExprKind::ArrayElement(array, index) => {
                let array_ptr = match self.eval_expr_as_readonly_ptr(array)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let index = match self.eval_expr(index)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let element_layout = ty.layout(self.module.ctx);
                let ptr_offset = self.cursor().arithmetic(
                    ArithmeticOp::Mul,
                    false,
                    index,
                    Value::new_i64(element_layout.size as i64),
                );
                EvalResult::Value(self.cursor().offset_ptr(array_ptr, Value::Definition(ptr_offset)))
            }
            ir_tree::ExprKind::ArrayInitializer(exprs) => {
                let alloca = Value::Definition(self.alloca(ty.layout(self.module.ctx)));
                let element_ty = ty.array_element_type().unwrap();
                self.eval_array_initializer_into(alloca, exprs, element_ty)?
            }
            ir_tree::ExprKind::StructInitializer(fields) => {
                let alloca = Value::Definition(self.alloca(ty.layout(self.module.ctx)));
                self.eval_struct_initializer_into(alloca, fields, &expr.ty)?
            }
            ir_tree::ExprKind::Load(place) => self.eval_place_as_ptr(place)?,
            _ => {
                let value = match self.eval_expr(expr)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let alloca = self.alloca(ty.layout(self.module.ctx));
                self.cursor().store(Value::Definition(alloca.clone()), value);
                EvalResult::Value(Value::Definition(alloca))
            }
        })
    }

    fn eval_const(&mut self, value: &ir_tree::Constant<'ctx>) -> Value {
        match &*value.with_erased_isize_usize(self.module.ctx) {
            ir_tree::Constant::Undefined(ty) => Value::Undefined(lower_type(self.module, *ty)),
            ir_tree::Constant::Null(_) => Value::Null,
            ir_tree::Constant::Unit => Value::Zst,
            ir_tree::Constant::Bool(bool) => Value::Bool(*bool),
            ir_tree::Constant::I8(int) => Value::Number {
                data: *int as i64,
                ty: Type::I8,
            },
            ir_tree::Constant::U8(int) => Value::Number {
                data: *int as i64,
                ty: Type::I8,
            },
            ir_tree::Constant::I32(int) => Value::Number {
                data: *int as i64,
                ty: Type::I32,
            },
            ir_tree::Constant::U32(int) => Value::Number {
                data: *int as i64,
                ty: Type::I32,
            },
            ir_tree::Constant::I64(int) => Value::Number {
                data: *int,
                ty: Type::I64,
            },
            ir_tree::Constant::U64(int) => Value::Number {
                data: *int as i64,
                ty: Type::I64,
            },
            ir_tree::Constant::ISize(_) | ir_tree::Constant::USize(_) => unreachable!(),
            ir_tree::Constant::Array(..) | ir_tree::Constant::Struct(..) => {
                let ty = lower_type(self.module, value.ty(self.module.ctx));
                let layout = ty.layout(self.module.ctx);
                let alloca = Value::Definition(self.alloca(layout));
                self.eval_const_into(alloca.clone(), value);
                Value::Definition(self.cursor().load(alloca, ty))
            }
        }
    }

    fn eval_const_into(&mut self, place_ptr: Value, value: &ir_tree::Constant<'ctx>) {
        match value {
            ir_tree::Constant::Undefined(_)
            | ir_tree::Constant::Null(_)
            | ir_tree::Constant::Unit
            | ir_tree::Constant::Bool(_)
            | ir_tree::Constant::I8(_)
            | ir_tree::Constant::U8(_)
            | ir_tree::Constant::I32(_)
            | ir_tree::Constant::U32(_)
            | ir_tree::Constant::I64(_)
            | ir_tree::Constant::U64(_)
            | ir_tree::Constant::ISize(_)
            | ir_tree::Constant::USize(_) => {
                let value = self.eval_const(value);
                self.cursor().store(place_ptr, value)
            }
            ir_tree::Constant::Array(ty, elements) => {
                let element_ty = ty.as_array().unwrap().0;
                let element_size = element_ty.layout(self.module.ctx).size;
                for (i, element) in elements.iter().enumerate() {
                    let ptr = self
                        .cursor()
                        .offset_ptr(place_ptr.clone(), Value::new_i64(i as i64 * element_size as i64));
                    self.eval_const_into(ptr, element);
                }
            }
            ir_tree::Constant::Struct(ty, fields) => {
                let (struct_, type_arguments) = ty.as_struct().unwrap();
                for ((_, field_id), field_value) in struct_.info().fields.iter().zip(fields) {
                    let offset = self.module.ctx.offset_of_struct_field(*field_id, type_arguments) as i64;
                    let ptr = self.cursor().offset_ptr(place_ptr.clone(), Value::new_i64(offset));
                    self.eval_const_into(ptr, field_value);
                }
            }
        }
    }

    /// Evaluate array initializer and store the elements into provided array place. Returns `place_ptr`.
    fn eval_array_initializer_into(
        &mut self,
        place_ptr: Value,
        exprs: &[ir_tree::Expr<'ctx>],
        element_ty: &Type,
    ) -> Result<EvalResult, Error> {
        let mut elements = Vec::new();
        for expr in exprs {
            match self.eval_expr(expr)? {
                EvalResult::Never => return Ok(EvalResult::Never),
                EvalResult::Value(element) => elements.push(element),
            }
        }

        let element_layout = element_ty.layout(self.module.ctx);
        for (i, element) in elements.into_iter().enumerate() {
            let ptr = self
                .cursor()
                .offset_ptr(place_ptr.clone(), Value::new_i64(i as i64 * element_layout.size as i64));
            self.cursor().store(ptr, element);
        }

        Ok(EvalResult::Value(place_ptr))
    }

    /// Evaluate struct initializer and store the fields into provided struct place. Returns `place_ptr`.
    fn eval_struct_initializer_into(
        &mut self,
        place_ptr: Value,
        fields: &[(ir_tree::StructFieldId, ir_tree::Expr<'ctx>)],
        struct_ty: &ir_tree::Type<'ctx>,
    ) -> Result<EvalResult, Error> {
        let mut exprs = Vec::new();
        let (_struct, type_arguments) = struct_ty.as_struct().unwrap();
        for (field_id, field_expr) in fields {
            match self.eval_expr(field_expr)? {
                EvalResult::Never => return Ok(EvalResult::Never),
                EvalResult::Value(val) => {
                    let offset = self.module.ctx.offset_of_struct_field(*field_id, type_arguments);
                    exprs.push((offset, val));
                }
            }
        }

        for (offset, value) in exprs {
            let ptr = self.cursor().offset_ptr(
                place_ptr.clone(),
                Value::Number {
                    data: offset as i64,
                    ty: Type::I64,
                },
            );
            self.cursor().store(ptr, value);
        }

        Ok(EvalResult::Value(place_ptr))
    }

    fn eval_place_as_ptr(&mut self, expr: &ir_tree::Place<'ctx>) -> Result<EvalResult, Error> {
        Ok(match &expr.kind {
            ir_tree::PlaceKind::Dereference(ptr) => self.eval_expr(ptr)?,
            ir_tree::PlaceKind::Variable(var_id) => {
                EvalResult::Value(Value::Definition(self.variable_map[var_id].clone()))
            }
            ir_tree::PlaceKind::Field(place, field_id) => {
                let (_struct, type_arguments) = place.ty.as_struct().unwrap();
                let field_offset = self.module.ctx.offset_of_struct_field(*field_id, type_arguments);
                let place_ptr = match self.eval_place_as_ptr(place)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                EvalResult::Value(self.cursor().offset_ptr(place_ptr, Value::new_i64(field_offset as i64)))
            }
            ir_tree::PlaceKind::ArrayElement(array, index) => {
                let array_ptr = match self.eval_place_as_ptr(array)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let index = match self.eval_expr(index)? {
                    EvalResult::Never => return Ok(EvalResult::Never),
                    EvalResult::Value(val) => val,
                };
                let element_ty = array.ty.as_array().unwrap().0;
                let element_size = element_ty.layout(self.module.ctx).size;
                let ptr_offset =
                    self.cursor()
                        .arithmetic(ArithmeticOp::Mul, false, index, Value::new_i64(element_size as i64));
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
