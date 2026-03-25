use super::*;
use crate::ast;

/// A memory layout of a type
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub size: u64,
    pub align: u64,
}

/// The set of data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    Never,
    Unit,
    Bool,
    Int(IntType),
    Struct(StructId),
    Ptr { pointee: Option<TypeId> },
    Array { element: TypeId, length: u64 },
}

impl Type {
    /// An opaque pointer type
    pub const OPAQUE_PTR: Self = Self::Ptr { pointee: None };

    /// A pointer to `i8` type
    pub const I8_PTR: Self = Self::Ptr {
        pointee: Some(TypeId::I8),
    };

    /// Wrap this type in a pointer
    pub fn make_ptr(self, typesystem: &mut TypeSystem) -> Self {
        Self::Ptr {
            pointee: Some(typesystem.get_type_id(self)),
        }
    }
}

/// Interegre data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntType {
    I8,
    U8,
    I32,
    U32,
    I64,
    U64,
}

impl IntType {
    /// Returns the number of bytes used to store this int
    pub fn bytes(self) -> u64 {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }

    /// Returns `true` if this data type is a signed integer
    pub fn is_signed(self) -> bool {
        match self {
            Self::I8 | Self::I32 | Self::I64 => true,
            Self::U8 | Self::U32 | Self::U64 => false,
        }
    }
}

/// The ID of a structure type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructId(usize);

/// A description of a structure
#[derive(Debug)]
pub struct Struct {
    pub name: ast::Ident,
    pub fields: Vec<StructField>,
}

/// A field of a struct definition
#[derive(Debug)]
pub struct StructField {
    pub name: ast::Ident,
    pub ty: Type,
}

/// The ID of a type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeId(usize);

impl TypeId {
    /// A well-known id of `i8`
    pub const I8: Self = Self(0);
}

/// Store the state of the type system
#[derive(Debug)]
pub struct TypeSystem {
    ptr_size: u64,
    structs: Vec<Struct>,
    types_with_ids: Vec<Type>,
    type_lut: HashMap<Type, TypeId>,
}

impl TypeSystem {
    /// Create a new type system context. Generally there should be only one context created.
    pub fn new(ptr_size: u64) -> Self {
        Self {
            ptr_size,
            structs: Vec::new(),
            types_with_ids: vec![Type::Int(IntType::I8)],
            type_lut: [(Type::Int(IntType::I8), TypeId::I8)].into_iter().collect(),
        }
    }

    /// Get or create a type ID for the given type
    pub fn get_type_id(&mut self, ty: Type) -> TypeId {
        if let Some(id) = self.type_lut.get(&ty).copied() {
            id
        } else {
            let id = TypeId(self.types_with_ids.len());
            self.types_with_ids.push(ty);
            self.type_lut.insert(ty, id);
            id
        }
    }

    /// Parse type from its AST representation
    pub fn type_from_ast(&mut self, type_namespace: &HashMap<String, Type>, ast: &ast::Type) -> Result<Type, Error> {
        match ast {
            ast::Type::Never(_) => Ok(Type::Never),
            ast::Type::Ident(ident) => type_namespace
                .get(&ident.value)
                .copied()
                .ok_or_else(|| Error::new(format!("unknown type {:?}", ident.value)).with_span(ident.span)),
            ast::Type::Ptr { star_span: _, pointee } => {
                let pointee = self.type_from_ast(type_namespace, pointee)?;
                let pointee_id = self.get_type_id(pointee);
                Ok(Type::Ptr {
                    pointee: Some(pointee_id),
                })
            }
            ast::Type::Array {
                element_type,
                length,
                left_bracket_span: _,
                right_bracket_span: _,
            } => {
                let element_type = self.type_from_ast(type_namespace, element_type)?;
                let element_type_id = self.get_type_id(element_type);
                let length = match &**length {
                    ast::Expr::Literal(ast::LiteralExpr {
                        span: _,
                        value: ast::LiteralExprValue::Number(num),
                    }) => *num as u64,
                    _ => return Err(Error::new("array length must be a number literal").with_span(length.span())),
                };
                Ok(Type::Array {
                    element: element_type_id,
                    length,
                })
            }
        }
    }

    /// Get a type of a struct definition from its AST representation
    pub fn struct_from_ast(
        &mut self,
        type_namespace: &HashMap<String, Type>,
        ast: &ast::Struct,
    ) -> Result<Type, Error> {
        let fields = ast
            .fields
            .iter()
            .map(|field| {
                Ok(StructField {
                    name: field.name.clone(),
                    ty: self.type_from_ast(type_namespace, &field.ty)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sid = StructId(self.structs.len());
        self.structs.push(Struct {
            name: ast.name.clone(),
            fields,
        });
        Ok(Type::Struct(sid))
    }

    /// Get the layout of a type
    pub fn layout_of(&self, ty: Type) -> Layout {
        match ty {
            Type::Never | Type::Unit => Layout { size: 0, align: 1 },
            Type::Bool => Layout { size: 1, align: 1 },
            Type::Int(i) => Layout {
                size: i.bytes(),
                align: i.bytes(),
            },
            Type::Ptr { pointee: _ } => Layout {
                size: self.ptr_size,
                align: self.ptr_size,
            },
            Type::Struct(sid) => {
                let s = &self.structs[sid.0];
                let mut layout = Layout { size: 0, align: 1 };
                for f in &s.fields {
                    let f_layout = self.layout_of(f.ty);
                    if f_layout.align > layout.align {
                        layout.align = f_layout.align;
                    }
                    layout.size = layout.size.next_multiple_of(f_layout.align);
                    layout.size += f_layout.size;
                }
                layout.size = layout.size.next_multiple_of(layout.align);
                layout
            }
            Type::Array { element, length } => {
                let mut layout = self.layout_of(self.get_type(element));
                layout.size = layout.size.next_multiple_of(layout.align);
                layout.size *= length;
                layout
            }
        }
    }

    /// Get struct field's byte offset and type
    pub fn get_struct_field(&self, sid: StructId, field: &ast::Ident) -> Result<(u64, Type), Error> {
        let s = &self.structs[sid.0];
        let mut offset = 0u64;
        for f in &s.fields {
            let f_layout = self.layout_of(f.ty);
            offset = offset.next_multiple_of(f_layout.align);
            if f.name.value == field.value {
                return Ok((offset, f.ty));
            }
            offset += f_layout.size;
        }
        Err(Error::new(format!("struct {} has no field {}", s.name.value, field.value)).with_span(field.span))
    }

    /// Get a referencse to the struct declaration
    pub fn get_struct(&self, sid: StructId) -> &Struct {
        &self.structs[sid.0]
    }

    /// Get the actual type by ID
    pub fn get_type(&self, id: TypeId) -> Type {
        self.types_with_ids[id.0]
    }
}

impl Type {
    /// Returns `true` if this data type is an integer
    pub fn is_int(self) -> bool {
        matches!(self, Self::Int(_))
    }
}
