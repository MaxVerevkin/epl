/// A basic arithmetic operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
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

/// A binary operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Cmp(CmpOp),
    Arithmetic(ArithmeticOp),
    LogicalOr,
    LogicalAnd,
}

/// A memory layout of a type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    pub size: u64,
    pub align: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PtrSize {
    _64,
}

impl PtrSize {
    pub fn bytes(self) -> u64 {
        match self {
            Self::_64 => 8,
        }
    }
}
