use super::*;

mod arithmetic_and_cmp;
mod const_abi;

#[derive(Debug)]
pub enum Error {
    Return(Constant),
    Break(LoopId, Constant),
    BadIr(String),
}

pub fn eval_comptime_expr(
    expr: &Expr,
    functions: &BTreeMap<FunctionId, Function>,
    typesystem: &TypeSystem,
) -> Result<Constant, Error> {
    let ExprKind::FunctionCall(function_id, args) = &expr.kind else {
        return Err(Error::BadIr(
            "only function calls are implemented inside comptime".to_owned(),
        ));
    };
    let args = args
        .iter()
        .map(|arg| eval_expr_simple(arg, functions, typesystem))
        .collect::<Result<Vec<_>, _>>()?;
    eval_pure_function(*function_id, functions, args, typesystem)
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

struct VariableMemory {
    bytes: Vec<u8>,
}

fn eval_pure_function(
    function_id: FunctionId,
    functions: &BTreeMap<FunctionId, Function>,
    arguments: Vec<Constant>,
    typesystem: &TypeSystem,
) -> Result<Constant, Error> {
    let function = &functions[&function_id];
    let body = function.body.as_ref().unwrap();
    let mut ctx = EvalCtx {
        function,
        functions,
        arguments,
        typesystem,
        variables: HashMap::new(),
    };
    match ctx.eval_expr(body) {
        Err(Error::Return(value)) => Ok(value),
        Err(Error::Break(..)) => unreachable!(),
        other => other,
    }
}

struct EvalCtx<'a> {
    function: &'a Function,
    functions: &'a BTreeMap<FunctionId, Function>,
    arguments: Vec<Constant>,
    typesystem: &'a TypeSystem,
    variables: HashMap<VariableId, VariableMemory>,
}

fn eval_expr_simple(
    expr: &Expr,
    functions: &BTreeMap<FunctionId, Function>,
    typesystem: &TypeSystem,
) -> Result<Constant, Error> {
    Ok(match &expr.kind {
        ExprKind::Const(value) => value.clone(),
        ExprKind::Comptime(expr) => eval_comptime_expr(expr, functions, typesystem)?,
        _other => return Err(Error::BadIr("this is not a simple expr".to_owned())),
    })
}

impl EvalCtx<'_> {
    fn load(&self, place: ConstantPlace, ty: Type) -> Constant {
        let layout = ty.layout(self.typesystem);
        let bytes = &self.variables[&place.variable].bytes[place.bytes_offset..][..layout.size as usize];
        const_abi::constant_from_bytes(bytes, ty, self.typesystem)
    }

    fn store(&mut self, place: ConstantPlace, value: &Constant) {
        let bytes = const_abi::constant_to_bytes(value, self.typesystem);
        self.variables.get_mut(&place.variable).unwrap().bytes[place.bytes_offset..][..bytes.len()]
            .copy_from_slice(&bytes);
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
                let i = self
                    .function
                    .args
                    .iter()
                    .position(|(name, _ty)| name == argument_name)
                    .unwrap();
                self.arguments[i].clone()
            }
            ExprKind::Block(block_expr) => {
                for (variable, ty) in &block_expr.variables {
                    let layout = ty.layout(self.typesystem);
                    self.variables.insert(
                        *variable,
                        VariableMemory {
                            bytes: vec![0; layout.size as usize],
                        },
                    );
                }
                fn on_block_exit(ctx: &mut EvalCtx<'_>, block_expr: &BlockExpr) {
                    for (variable, _) in &block_expr.variables {
                        ctx.variables.remove(variable);
                    }
                }
                let value = 'blk: {
                    for (expr_i, expr) in block_expr.exprs.iter().enumerate() {
                        match self.eval_expr(expr) {
                            Err(e) => {
                                on_block_exit(self, block_expr);
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
                on_block_exit(self, block_expr);
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
                assert!(self.functions[function_id].is_pure, "can only call pure functions");
                let arguments_values = arguments
                    .iter()
                    .map(|expr| self.eval_expr(expr))
                    .collect::<Result<Vec<_>, _>>()?;
                eval_pure_function(*function_id, self.functions, arguments_values, self.typesystem)?
            }
            ExprKind::Cast(expr_to_cast) => arithmetic_and_cmp::eval_cast(self.eval_expr(expr_to_cast)?, expr.ty)?,
            ExprKind::Not(expr) => Constant::Bool(!self.eval_expr(expr)?.into_bool().unwrap()),
            ExprKind::Comptime(expr) => eval_comptime_expr(expr, self.functions, self.typesystem)?,
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
                    .array_element_type(self.typesystem)
                    .unwrap()
                    .layout(self.typesystem)
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
