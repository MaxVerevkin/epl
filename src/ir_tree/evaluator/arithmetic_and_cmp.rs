use super::*;

pub fn eval_arithmetic<'ctx>(
    ctx: Context<'ctx>,
    op: ArithmeticOp,
    lhs: Constant<'ctx>,
    rhs: Constant<'ctx>,
) -> Result<Constant<'ctx>, Error> {
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
        (Constant::ISize(lhs), Constant::ISize(rhs)) => match ctx.ptr_size() {
            PtrSize::_64 => Constant::ISize(const_arithmetic_i64(op, lhs as _, rhs as _) as _),
        },
        (Constant::USize(lhs), Constant::USize(rhs)) => match ctx.ptr_size() {
            PtrSize::_64 => Constant::USize(const_arithmetic_u64(op, lhs as _, rhs as _) as _),
        },
        (lhs, rhs) => {
            return Err(Error::new(format!(
                "arithmetic: unsupported operation {lhs:?} {op:?} {rhs:?}"
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
                "comparison: unsupported operation {lhs:?} {op:?} {rhs:?}"
            )));
        }
    })
}

pub fn eval_cast<'ctx>(
    ctx: Context<'ctx>,
    from: Constant<'ctx>,
    target_ty: Type<'ctx>,
) -> Result<Constant<'ctx>, Error> {
    macro_rules! make_const_int_cast {
        ($($name:ident, $ty:ty;)*) => {
            $(
                fn $name(from: $ty, target_ty: Type, ptr_size: PtrSize) -> Constant {
                    match target_ty.info() {
                        TypeInfo::Never | TypeInfo::Unit | TypeInfo::Struct { .. } | TypeInfo::Array { .. } | TypeInfo::Ptr { .. } | TypeInfo::Bool | TypeInfo::TypeParameter { .. } => {
                            unreachable!()
                        }
                        TypeInfo::Int(int_type) => match int_type {
                            IntType::I8 => Constant::I8(from as _),
                            IntType::U8 => Constant::U8(from as _),
                            IntType::I32 => Constant::I32(from as _),
                            IntType::U32 => Constant::U32(from as _),
                            IntType::I64 => Constant::I64(from as _),
                            IntType::U64 => Constant::U64(from as _),
                            IntType::ISize => match ptr_size {
                                PtrSize::_64 => Constant::ISize(from as i64 as _),
                            }
                            IntType::USize => match ptr_size {
                                PtrSize::_64 => Constant::USize(from as u64 as _),
                            }
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

    let ptr_size = ctx.ptr_size();

    Ok(match from {
        Constant::Undefined(_) => todo!(),
        Constant::Null(_) => todo!(),
        Constant::Unit => unreachable!(),
        Constant::Bool(_) => unreachable!(),
        Constant::I8(int) => const_cast_i8(int, target_ty, ptr_size),
        Constant::U8(int) => const_cast_u8(int, target_ty, ptr_size),
        Constant::I32(int) => const_cast_i32(int, target_ty, ptr_size),
        Constant::U32(int) => const_cast_u32(int, target_ty, ptr_size),
        Constant::I64(int) => const_cast_i64(int, target_ty, ptr_size),
        Constant::U64(int) => const_cast_u64(int, target_ty, ptr_size),
        Constant::ISize(int) => match ctx.ptr_size() {
            PtrSize::_64 => const_cast_i64(int as _, target_ty, ptr_size),
        },
        Constant::USize(int) => match ctx.ptr_size() {
            PtrSize::_64 => const_cast_u64(int as _, target_ty, ptr_size),
        },
        Constant::Array(..) => unreachable!(),
        Constant::Struct(..) => unreachable!(),
    })
}
