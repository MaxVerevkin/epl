use super::*;

fn constant_to_bytes_into(constant: &Constant, typesystem: &TypeSystem, output: &mut Vec<u8>) {
    match constant {
        Constant::Undefined(ty) => {
            for _ in 0..ty.layout(typesystem).size {
                output.push(0);
            }
        }
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
                constant_to_bytes_into(element, typesystem, output);
            }
        }
        Constant::Struct(struct_id, fields) => {
            let struct_type = typesystem.get_struct(*struct_id);
            let mut written = 0;
            for (field_def, field_value) in struct_type.fields.iter().zip(fields) {
                while written < field_def.offset {
                    // padding
                    output.push(0);
                    written += 1;
                }
                constant_to_bytes_into(field_value, typesystem, output);
                written += field_def.ty.layout(typesystem).size;
            }
            for _ in written..struct_type.layout.size {
                // padding
                output.push(0);
            }
        }
    }
}

pub fn constant_to_bytes(constant: &Constant, typesystem: &TypeSystem) -> Vec<u8> {
    let mut output = Vec::new();
    constant_to_bytes_into(constant, typesystem, &mut output);
    output
}

pub fn constant_from_bytes(bytes: &[u8], ty: Type, typesystem: &TypeSystem) -> Constant {
    match ty {
        Type::Never => unreachable!(),
        Type::Unit => Constant::Unit,
        Type::Bool => Constant::Bool(bytes[0] == 1),
        Type::Int(int_type) => match int_type {
            IntType::I8 => Constant::I8(bytes[0] as i8),
            IntType::U8 => Constant::U8(bytes[0]),
            IntType::I32 => Constant::I32(i32::from_le_bytes(bytes.try_into().unwrap())),
            IntType::U32 => Constant::U32(u32::from_le_bytes(bytes.try_into().unwrap())),
            IntType::I64 => Constant::I64(i64::from_le_bytes(bytes.try_into().unwrap())),
            IntType::U64 => Constant::U64(u64::from_le_bytes(bytes.try_into().unwrap())),
        },
        Type::Struct(struct_id) => {
            let mut fields = Vec::new();
            let struct_type = typesystem.get_struct(struct_id);
            assert_eq!(struct_type.layout.size, bytes.len() as u64);
            for field_def in &struct_type.fields {
                let field_size = field_def.ty.layout(typesystem).size;
                let field_bytes = &bytes[field_def.offset as usize..][..field_size as usize];
                fields.push(constant_from_bytes(field_bytes, field_def.ty, typesystem));
            }
            Constant::Struct(struct_id, fields)
        }
        Type::Ptr { .. } => panic!("reading pointers is not a pure operation"),
        Type::Array { element, length } => {
            let mut elements = Vec::new();
            let mut bytes = bytes;
            let element_ty = typesystem.get_type(element);
            let element_size = element_ty.layout(typesystem).size as usize;
            for _ in 0..length {
                let (element_bytes, rest_bytes) = bytes.split_at(element_size);
                elements.push(constant_from_bytes(element_bytes, element_ty, typesystem));
                bytes = rest_bytes;
            }
            Constant::Array(element, elements)
        }
    }
}
