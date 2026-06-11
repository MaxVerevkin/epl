use super::*;

fn constant_to_bytes_into<'ctx>(ctx: Context<'ctx>, constant: &Constant<'ctx>, output: &mut Vec<u8>) {
    match constant {
        Constant::Undefined(ty) => {
            for _ in 0..ty.layout(ctx).size {
                output.push(0);
            }
        }
        Constant::Null(_) => unimplemented!(),
        Constant::Unit => (),
        Constant::Bool(x) => output.push(*x as u8),
        Constant::I8(x) => output.push(*x as u8),
        Constant::U8(x) => output.push(*x),
        Constant::I32(x) => output.extend_from_slice(&x.to_le_bytes()),
        Constant::U32(x) => output.extend_from_slice(&x.to_le_bytes()),
        Constant::I64(x) => output.extend_from_slice(&x.to_le_bytes()),
        Constant::U64(x) => output.extend_from_slice(&x.to_le_bytes()),
        Constant::Array(_, elements) => {
            for element in elements {
                constant_to_bytes_into(ctx, element, output);
            }
        }
        Constant::Struct(ty, fields) => {
            let (struct_, type_arguments) = match ty.info() {
                TypeInfo::Struct {
                    struct_,
                    type_arguments,
                } => (*struct_, type_arguments),
                _ => unreachable!(),
            };
            let mut written = 0;
            for ((_, field_def_id), field_value) in struct_.info().fields.iter().zip(fields) {
                let offset = ctx.offset_of_struct_field(*field_def_id, type_arguments);
                while written < offset {
                    // padding
                    output.push(0);
                    written += 1;
                }
                constant_to_bytes_into(ctx, field_value, output);
                written += ctx.type_of_struct_field(*field_def_id, type_arguments).layout(ctx).size;
            }
            for _ in written..ty.layout(ctx).size {
                // padding
                output.push(0);
            }
        }
    }
}

pub fn constant_to_bytes<'ctx>(ctx: Context<'ctx>, constant: &Constant<'ctx>) -> Vec<u8> {
    let mut output = Vec::new();
    constant_to_bytes_into(ctx, constant, &mut output);
    output
}

pub fn constant_from_bytes<'ctx>(ctx: Context<'ctx>, bytes: &[u8], ty: Type<'ctx>) -> Constant<'ctx> {
    match ty.info() {
        TypeInfo::Never | TypeInfo::TypeParameter { .. } => unreachable!(),
        TypeInfo::Unit => Constant::Unit,
        TypeInfo::Bool => Constant::Bool(bytes[0] == 1),
        TypeInfo::Int(int_type) => match int_type {
            IntType::I8 => Constant::I8(bytes[0] as i8),
            IntType::U8 => Constant::U8(bytes[0]),
            IntType::I32 => Constant::I32(i32::from_le_bytes(bytes.try_into().unwrap())),
            IntType::U32 => Constant::U32(u32::from_le_bytes(bytes.try_into().unwrap())),
            IntType::I64 => Constant::I64(i64::from_le_bytes(bytes.try_into().unwrap())),
            IntType::U64 => Constant::U64(u64::from_le_bytes(bytes.try_into().unwrap())),
        },
        TypeInfo::Struct {
            struct_,
            type_arguments,
        } => {
            let mut fields = Vec::new();
            assert_eq!(ty.layout(ctx).size, bytes.len() as u64);
            for (_, field_def_id) in &struct_.info().fields {
                let field_ty = ctx.type_of_struct_field(*field_def_id, type_arguments);
                let field_size = field_ty.layout(ctx).size;
                let field_offset = ctx.offset_of_struct_field(*field_def_id, type_arguments);
                let field_bytes = &bytes[field_offset as usize..][..field_size as usize];
                fields.push(constant_from_bytes(ctx, field_bytes, field_ty));
            }
            Constant::Struct(ty, fields)
        }
        TypeInfo::Ptr { .. } => panic!("reading pointers is not a pure operation"),
        TypeInfo::Array { element_ty, length } => {
            let mut elements = Vec::with_capacity(*length as usize);
            let mut bytes = bytes;
            let element_size = element_ty.layout(ctx).size as usize;
            for _ in 0..*length {
                let (element_bytes, rest_bytes) = bytes.split_at(element_size);
                elements.push(constant_from_bytes(ctx, element_bytes, *element_ty));
                bytes = rest_bytes;
            }
            Constant::Array(ty, elements)
        }
    }
}
