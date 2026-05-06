use super::*;

mod arithmetic_and_cmp;
mod const_abi;

#[derive(Debug)]
enum EvalError {
    Return(Constant),
    Break(LoopId, Constant),
    Continue(LoopId),
    Error(Error),
}

impl From<Error> for EvalError {
    fn from(value: Error) -> Self {
        Self::Error(value)
    }
}

pub fn eval_comptime_expr(expr: &Expr, module: &Module) -> Result<Constant, Error> {
    let mut ctx = EvalCtx {
        module,
        in_function_call: None,
        scope: Scope::default(),
    };
    match ctx.eval_expr(expr) {
        Err(EvalError::Return(..)) => panic!("cannot return from a comptime block"),
        Err(EvalError::Break(..)) => panic!("cannot break from a comptime block"),
        Err(EvalError::Continue(..)) => panic!("cannot continue from a comptime block"),
        Err(EvalError::Error(err)) => Err(err),
        Ok(ok) => Ok(ok),
    }
}

fn eval_pure_function(function_id: FunctionId, arguments: &[Constant], module: &Module) -> Result<Constant, Error> {
    let function = &module.functions[&function_id];
    let body = function.body.as_ref().unwrap();
    let mut ctx = EvalCtx {
        module,
        in_function_call: Some(InFunctionCall { arguments }),
        scope: Scope::default(),
    };
    match ctx.eval_expr(body) {
        Err(EvalError::Return(value)) => Ok(value),
        Err(EvalError::Break(..) | EvalError::Continue(..)) => unreachable!(),
        Err(EvalError::Error(err)) => Err(err),
        Ok(ok) => Ok(ok),
    }
}

impl Constant {
    pub fn into_bool(self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ConstantPlace {
    variable: VariableId,
    bytes_offset: usize,
}

#[derive(Debug)]
struct VariableMemory {
    bytes: Vec<u8>,
}

struct EvalCtx<'a> {
    module: &'a Module,
    in_function_call: Option<InFunctionCall<'a>>,
    scope: Scope,
}

#[derive(Clone, Copy)]
struct InFunctionCall<'a> {
    arguments: &'a [Constant],
}

#[derive(Default, Debug)]
struct Scope {
    variables: HashMap<VariableId, VariableMemory>,
    parent: Option<Box<Self>>,
}

impl Scope {
    fn get_variable(&mut self, id: VariableId) -> Option<&mut VariableMemory> {
        self.variables
            .get_mut(&id)
            .or_else(|| self.parent.as_mut().and_then(|p| p.get_variable(id)))
    }
}

impl EvalCtx<'_> {
    fn load(&mut self, place: ConstantPlace, ty: Type) -> Constant {
        let mem = self.scope.get_variable(place.variable).unwrap();
        let layout = ty.layout(&self.module.typesystem);
        let bytes = &mem.bytes[place.bytes_offset..][..layout.size as usize];
        const_abi::constant_from_bytes(bytes, ty, &self.module.typesystem)
    }

    fn store(&mut self, place: ConstantPlace, value: &Constant) {
        let mem = self.scope.get_variable(place.variable).unwrap();
        let bytes = const_abi::constant_to_bytes(value, &self.module.typesystem);
        mem.bytes[place.bytes_offset..][..bytes.len()].copy_from_slice(&bytes);
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Constant, EvalError> {
        Ok(match &expr.kind {
            ExprKind::Const(value) => value.clone(),
            ExprKind::ConstString(_) => panic!("constant strings are not pure operations"),
            ExprKind::Load(place) => {
                let place_value = self.eval_place(place)?;
                self.load(place_value, expr.ty)
            }
            ExprKind::Field(struct_expr, field_name) => {
                let (struct_id, mut fields) = match self.eval_expr(struct_expr)? {
                    Constant::Struct(struct_id, fields) => (struct_id, fields),
                    _ => unreachable!(),
                };
                let field_index = self
                    .module
                    .typesystem
                    .get_struct(struct_id)
                    .fields
                    .iter()
                    .position(|f| f.name.value == *field_name)
                    .unwrap();
                fields.remove(field_index)
            }
            ExprKind::ArrayElement(expr, index) => {
                let mut elements = match self.eval_expr(expr)? {
                    Constant::Array(_, elements) => elements,
                    _other => unreachable!(),
                };
                let index_value = match self.eval_expr(index)? {
                    Constant::U64(index) => index,
                    _other => unreachable!(),
                };
                elements.remove(index_value as usize)
            }
            ExprKind::Store(place, expr) => {
                let place_value = self.eval_place(place)?;
                let expr_value = self.eval_expr(expr)?;
                self.store(place_value, &expr_value);
                Constant::Unit
            }
            ExprKind::GetPointer(_) => panic!("getting pointers is not a pure operation"),
            ExprKind::Argument(arg_index) => self.in_function_call.as_ref().unwrap().arguments[*arg_index].clone(),
            ExprKind::Block(block_expr) => {
                let mut ctx = EvalCtx {
                    module: self.module,
                    in_function_call: self.in_function_call,
                    scope: Scope {
                        variables: block_expr
                            .variables
                            .iter()
                            .map(|decl| {
                                (
                                    decl.id,
                                    VariableMemory {
                                        bytes: vec![0; decl.ty.layout(&self.module.typesystem).size as usize],
                                    },
                                )
                            })
                            .collect(),
                        parent: Some(Box::new(std::mem::take(&mut self.scope))),
                    },
                };
                let value = 'blk: {
                    for (expr_i, expr) in block_expr.exprs.iter().enumerate() {
                        match ctx.eval_expr(expr) {
                            Err(e) => {
                                self.scope = *ctx.scope.parent.unwrap();
                                return Err(e);
                            }
                            Ok(value) => {
                                if expr_i + 1 == block_expr.exprs.len() {
                                    break 'blk value;
                                }
                            }
                        }
                    }
                    Constant::Unit
                };
                self.scope = *ctx.scope.parent.unwrap();
                value
            }
            ExprKind::Return(expr) => return Err(EvalError::Return(self.eval_expr(expr)?)),
            ExprKind::Break(loop_id, expr) => return Err(EvalError::Break(*loop_id, self.eval_expr(expr)?)),
            ExprKind::Continue(loop_id) => return Err(EvalError::Continue(*loop_id)),
            ExprKind::Arithmetic(op, lhs, rhs) => {
                let lhs_value = self.eval_expr(lhs)?;
                let rhs_value = self.eval_expr(rhs)?;
                arithmetic_and_cmp::eval_arithmetic(*op, lhs_value, rhs_value)?
            }
            ExprKind::InPlaceArithmetic(op, place, expr) => {
                let place_value = self.eval_place(place)?;
                let expr_value = self.eval_expr(expr)?;
                let result = arithmetic_and_cmp::eval_arithmetic(*op, self.load(place_value, expr.ty), expr_value)?;
                self.store(place_value, &result);
                Constant::Unit
            }
            ExprKind::Cmp(op, lhs, rhs) => {
                let lhs_value = self.eval_expr(lhs)?;
                let rhs_value = self.eval_expr(rhs)?;
                Constant::Bool(arithmetic_and_cmp::eval_cmp(*op, lhs_value, rhs_value)?)
            }
            ExprKind::If {
                cond,
                if_true,
                if_false,
            } => match self.eval_expr(cond)?.into_bool().unwrap() {
                true => self.eval_expr(if_true)?,
                false => self.eval_expr(if_false)?,
            },
            ExprKind::Loop(loop_id, body) => loop {
                match self.eval_expr(body) {
                    Err(EvalError::Break(break_loop_id, break_value)) if break_loop_id == *loop_id => {
                        break break_value;
                    }
                    Err(EvalError::Continue(continue_loop_id)) if continue_loop_id == *loop_id => (),
                    Err(e) => return Err(e),
                    Ok(_) => (),
                }
            },
            ExprKind::ArrayInitializer(elements_exprs) => {
                let elements = elements_exprs
                    .iter()
                    .map(|expr| self.eval_expr(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                Constant::Array(expr.ty.array_element_type_id().unwrap(), elements)
            }
            ExprKind::StructInitializer(fields_exprs) => {
                let mut fields_constants = HashMap::new();
                for (field_name, field_expr) in fields_exprs {
                    fields_constants.insert(&**field_name, self.eval_expr(field_expr)?);
                }
                let mut fields_in_order = Vec::new();
                let struct_id = expr.ty.as_struct().unwrap();
                for field_def in &self.module.typesystem.get_struct(struct_id).fields {
                    fields_in_order.push(fields_constants.remove(&*field_def.name.value).unwrap());
                }
                Constant::Struct(struct_id, fields_in_order)
            }
            ExprKind::FunctionCall(function_id, arguments) => {
                assert!(
                    self.module.functions[function_id].is_pure,
                    "can only call pure functions"
                );
                let arguments_values = arguments
                    .iter()
                    .map(|expr| self.eval_expr(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                eval_pure_function(*function_id, &arguments_values, self.module)?
            }
            ExprKind::Cast(expr_to_cast) => arithmetic_and_cmp::eval_cast(self.eval_expr(expr_to_cast)?, expr.ty)?,
            ExprKind::Not(expr) => Constant::Bool(!self.eval_expr(expr)?.into_bool().unwrap()),
            ExprKind::Comptime(expr) => self.eval_expr(expr)?,
        })
    }

    fn eval_place(&mut self, place_expr: &Place) -> Result<ConstantPlace, EvalError> {
        Ok(match &place_expr.kind {
            PlaceKind::Dereference(_) => panic!("dereference is not a pure operation"),
            PlaceKind::Variable(variable_id) => ConstantPlace {
                variable: *variable_id,
                bytes_offset: 0,
            },
            PlaceKind::Field(struct_place, field) => {
                let struct_place_value = self.eval_place(struct_place)?;
                let field_offset = self
                    .module
                    .typesystem
                    .get_struct(struct_place.ty.as_struct().unwrap())
                    .fields
                    .iter()
                    .find(|f| f.name.value == *field)
                    .unwrap()
                    .offset;
                ConstantPlace {
                    variable: struct_place_value.variable,
                    bytes_offset: struct_place_value.bytes_offset + field_offset as usize,
                }
            }
            PlaceKind::ArrayElement(array_place, index_expr) => {
                let array_place_value = self.eval_place(array_place)?;
                let index_value = self.eval_expr(index_expr)?;
                let element_size = array_place
                    .ty
                    .array_element_type(&self.module.typesystem)
                    .unwrap()
                    .layout(&self.module.typesystem)
                    .size;
                let offset = match index_value {
                    Constant::U64(index) => index * element_size,
                    _ => panic!("index value must be of type u64"),
                };
                ConstantPlace {
                    variable: array_place_value.variable,
                    bytes_offset: array_place_value.bytes_offset + offset as usize,
                }
            }
        })
    }
}
