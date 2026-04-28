use super::*;

mod arithmetic_and_cmp;
mod const_abi;

#[derive(Debug)]
pub enum Error {
    Return(Constant),
    Break(LoopId, Constant),
    BadIr(String),
}

pub fn eval_comptime_expr(expr: &Expr, module: &Module) -> Result<Constant, Error> {
    let mut ctx = EvalCtx {
        module,
        in_function_call: None,
        scope: Scope::default(),
    };
    match ctx.eval_expr(expr) {
        Err(Error::Return(..)) => panic!("cannot return from a comptime block"),
        Err(Error::Break(..)) => panic!("cannot break from a comptime block"),
        other => other,
    }
}

fn eval_pure_function(function_id: FunctionId, arguments: &[Constant], module: &Module) -> Result<Constant, Error> {
    let function = &module.functions[&function_id];
    let body = function.body.as_ref().unwrap();
    let mut ctx = EvalCtx {
        module,
        in_function_call: Some(InFunctionCall { function, arguments }),
        scope: Scope::default(),
    };
    match ctx.eval_expr(body) {
        Err(Error::Return(value)) => Ok(value),
        Err(Error::Break(..)) => unreachable!(),
        other => other,
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
    function: &'a Function,
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

    fn eval_expr(&mut self, expr: &Expr) -> Result<Constant, Error> {
        Ok(match &expr.kind {
            ExprKind::Const(value) => value.clone(),
            ExprKind::ConstString(s) => todo!(),
            ExprKind::Load(place) => {
                let place_value = self.eval_place(place)?;
                self.load(place_value, expr.ty)
            }
            ExprKind::Field(expr, field_name) => {
                let struct_value = self.eval_expr(expr)?;
                todo!()
            }
            ExprKind::ArrayElement(expr, index) => {
                let array_value = self.eval_expr(expr)?;
                let index_value = self.eval_expr(expr)?;
                todo!()
            }
            ExprKind::Store(place, expr) => {
                let place_value = self.eval_place(place)?;
                let expr_value = self.eval_expr(expr)?;
                self.store(place_value, &expr_value);
                Constant::Unit
            }
            ExprKind::GetPointer(place) => todo!(),
            ExprKind::Argument(argument_name) => {
                let in_fn = self.in_function_call.as_ref().unwrap();
                let i = in_fn
                    .function
                    .args
                    .iter()
                    .position(|(name, _ty)| name == argument_name)
                    .unwrap();
                in_fn.arguments[i].clone()
            }
            ExprKind::Block(block_expr) => {
                let mut ctx = EvalCtx {
                    module: self.module,
                    in_function_call: self.in_function_call,
                    scope: Scope {
                        variables: block_expr
                            .variables
                            .iter()
                            .map(|(id, ty)| {
                                (
                                    *id,
                                    VariableMemory {
                                        bytes: vec![0; ty.layout(&self.module.typesystem).size as usize],
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
            ExprKind::Return(expr) => return Err(Error::Return(self.eval_expr(expr)?)),
            ExprKind::Break(loop_id, expr) => return Err(Error::Break(*loop_id, self.eval_expr(expr)?)),
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
                    Err(Error::Break(break_loop_id, break_value)) if break_loop_id == *loop_id => break break_value,
                    Err(e) => return Err(e),
                    Ok(_) => (),
                }
            },
            ExprKind::ArrayInitializer(exprs) => todo!(),
            ExprKind::StructInitializer(items) => todo!(),
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

    fn eval_place(&mut self, place_expr: &Place) -> Result<ConstantPlace, Error> {
        Ok(match &place_expr.kind {
            PlaceKind::Dereference(expr) => panic!("dereference is not a pure operation"),
            PlaceKind::Variable(variable_id) => ConstantPlace {
                variable: *variable_id,
                bytes_offset: 0,
            },
            PlaceKind::Field(place, _) => todo!(),
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
