/// A basic arithmetic operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A basic comparison operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    Equal,
    NotEqual,
}
