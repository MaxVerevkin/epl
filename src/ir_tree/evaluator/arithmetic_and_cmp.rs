use super::*;

pub fn eval_arithmetic(op: ArithmeticOp, lhs: Constant, rhs: Constant) -> Result<Constant, Error> {
    macro_rules! make_const_arithmetic {
        ($($name:ident, $ty:ty;)*) => {
            $(
                fn $name(op: ArithmeticOp, lhs: $ty, rhs: $ty) -> $ty {
                    match op {
                        ArithmeticOp::Add => lhs.wrapping_add(rhs),
                        ArithmeticOp::Sub => lhs.wrapping_sub(rhs),
                        ArithmeticOp::Mul => lhs.wrapping_mul(rhs),
                        ArithmeticOp::Div => lhs.wrapping_div(rhs),
                        ArithmeticOp::Rem => lhs.wrapping_rem(rhs),
                    }
                }
            )*
        };
    }

    make_const_arithmetic! {
        const_arithmetic_i8, i8;
        const_arithmetic_u8, u8;
        const_arithmetic_i32, i32;
        const_arithmetic_u32, u32;
        const_arithmetic_i64, i64;
        const_arithmetic_u64, u64;
    }

    Ok(match (lhs, rhs) {
        (Constant::I8(lhs), Constant::I8(rhs)) => Constant::I8(const_arithmetic_i8(op, lhs, rhs)),
        (Constant::U8(lhs), Constant::U8(rhs)) => Constant::U8(const_arithmetic_u8(op, lhs, rhs)),
        (Constant::I32(lhs), Constant::I32(rhs)) => Constant::I32(const_arithmetic_i32(op, lhs, rhs)),
        (Constant::U32(lhs), Constant::U32(rhs)) => Constant::U32(const_arithmetic_u32(op, lhs, rhs)),
        (Constant::I64(lhs), Constant::I64(rhs)) => Constant::I64(const_arithmetic_i64(op, lhs, rhs)),
        (Constant::U64(lhs), Constant::U64(rhs)) => Constant::U64(const_arithmetic_u64(op, lhs, rhs)),
        (lhs, rhs) => {
            return Err(Error::new(format!(
                "arithmetic: unuspported operation {lhs:?} {op:?} {rhs:?}"
            )));
        }
    })
}

pub fn eval_cmp(op: CmpOp, lhs: Constant, rhs: Constant) -> Result<bool, Error> {
    macro_rules! make_const_cmp {
        ($($name:ident, $ty:ty;)*) => {
            $(
                fn $name(op: CmpOp, lhs: $ty, rhs: $ty) -> bool {
                    match op {
                        CmpOp::Less => lhs < rhs,
                        CmpOp::LessOrEqual => lhs <= rhs,
                        CmpOp::Greater => lhs > rhs,
                        CmpOp::GreaterOrEqual => lhs >= rhs,
                        CmpOp::Equal => lhs == rhs,
                        CmpOp::NotEqual => lhs != rhs,
                    }
                }
            )*
        };
    }

    make_const_cmp! {
        const_cmp_bool, bool;
        const_cmp_i8, i8;
        const_cmp_u8, u8;
        const_cmp_i32, i32;
        const_cmp_u32, u32;
        const_cmp_i64, i64;
        const_cmp_u64, u64;
    }

    Ok(match (lhs, rhs) {
        (Constant::Bool(lhs), Constant::Bool(rhs)) => const_cmp_bool(op, lhs, rhs),
        (Constant::I8(lhs), Constant::I8(rhs)) => const_cmp_i8(op, lhs, rhs),
        (Constant::U8(lhs), Constant::U8(rhs)) => const_cmp_u8(op, lhs, rhs),
        (Constant::I32(lhs), Constant::I32(rhs)) => const_cmp_i32(op, lhs, rhs),
        (Constant::U32(lhs), Constant::U32(rhs)) => const_cmp_u32(op, lhs, rhs),
        (Constant::I64(lhs), Constant::I64(rhs)) => const_cmp_i64(op, lhs, rhs),
        (Constant::U64(lhs), Constant::U64(rhs)) => const_cmp_u64(op, lhs, rhs),
        (lhs, rhs) => {
            return Err(Error::new(format!(
                "comparison: unuspported operation {lhs:?} {op:?} {rhs:?}"
            )));
        }
    })
}

pub fn eval_cast(from: Constant, target_ty: Type) -> Result<Constant, Error> {
    macro_rules! make_const_int_cast {
        ($($name:ident, $ty:ty;)*) => {
            $(
                fn $name(from: $ty, target_ty: Type) -> Constant {
                    match target_ty {
                        Type::Never | Type::Unit | Type::Struct(_) | Type::Array { .. } | Type::Ptr { .. } | Type::Bool => {
                            unreachable!()
                        }
                        Type::Int(int_type) => match int_type {
                            IntType::I8 => Constant::I8(from as _),
                            IntType::U8 => Constant::U8(from as _),
                            IntType::I32 => Constant::I32(from as _),
                            IntType::U32 => Constant::U32(from as _),
                            IntType::I64 => Constant::I64(from as _),
                            IntType::U64 => Constant::U64(from as _),
                        },
                    }
                }
            )*
        };
    }

    make_const_int_cast! {
        const_cast_i8, i8;
        const_cast_u8, u8;
        const_cast_i32, i32;
        const_cast_u32, u32;
        const_cast_i64, i64;
        const_cast_u64, u64;
    }

    Ok(match from {
        Constant::Undefined(_) => todo!(),
        Constant::Unit => unreachable!(),
        Constant::Bool(_) => unreachable!(),
        Constant::I8(int) => const_cast_i8(int, target_ty),
        Constant::U8(int) => const_cast_u8(int, target_ty),
        Constant::I32(int) => const_cast_i32(int, target_ty),
        Constant::U32(int) => const_cast_u32(int, target_ty),
        Constant::I64(int) => const_cast_i64(int, target_ty),
        Constant::U64(int) => const_cast_u64(int, target_ty),
        Constant::Array(..) => unreachable!(),
        Constant::Struct(..) => unreachable!(),
    })
}
