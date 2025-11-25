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
    Void,
    Bool,
    Int(IntType),
    Struct(StructId),
    Ptr(PtrId),
}

/// Interegre data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntType {
    I8,
    U8,
    I32,
    U32,
}

impl IntType {
    /// Returns the number of bits used to store this int
    pub fn bits(self) -> u64 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I32 | Self::U32 => 32,
        }
    }

    /// Returns the number of bytes used to store this int
    pub fn bytes(self) -> u64 {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I32 | Self::U32 => 4,
        }
    }

    /// Returns `true` if this data type is a signed integer
    pub fn is_signed(self) -> bool {
        match self {
            Self::I8 | Self::I32 => true,
            Self::U8 | Self::U32 => false,
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

/// The ID of a pointer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PtrId(usize);

impl PtrId {
    /// A well-known opaque pointer
    pub const OPAQUE: Self = Self(0);

    /// A well-known pointer `*i8`
    pub const TO_I8: Self = Self(1);
}

/// A description of a pointer
#[derive(Debug)]
pub struct Ptr {
    pub pointee: Option<Type>,
}

/// Store the state of the type system
#[derive(Debug)]
pub struct TypeSystem {
    ptr_size: u64,
    structs: Vec<Struct>,
    pointers: Vec<Ptr>,
    pointer_lut: HashMap<Option<Type>, PtrId>, // pointee -> poniter id
}

impl TypeSystem {
    /// Create a new type system context. Generally there should be only one context created.
    pub fn new(ptr_size: u64) -> Self {
        Self {
            ptr_size,
            structs: Vec::new(),
            pointers: vec![
                Ptr { pointee: None },
                Ptr {
                    pointee: Some(Type::Int(IntType::I8)),
                },
            ],
            pointer_lut: [(None, PtrId::OPAQUE), (Some(Type::Int(IntType::I8)), PtrId::TO_I8)]
                .into_iter()
                .collect(),
        }
    }

    /// Get a pointer ID given the pointee, or allocate and cache a new one
    pub fn get_or_create_pointer_type(&mut self, pointee: Option<Type>) -> PtrId {
        if let Some(pid) = self.pointer_lut.get(&pointee).copied() {
            pid
        } else {
            let pid = PtrId(self.pointers.len());
            self.pointers.push(Ptr { pointee });
            self.pointer_lut.insert(pointee, pid);
            pid
        }
    }

    /// Parse type from its AST representation
    pub fn type_from_ast(&mut self, type_namespace: &HashMap<String, Type>, ast: &ast::Type) -> Result<Type, Error> {
        match &ast.value {
            ast::TypeValue::Never => Ok(Type::Never),
            ast::TypeValue::Ident(name) => type_namespace
                .get(name)
                .copied()
                .ok_or_else(|| Error::new(format!("unknown type {name:?}")).with_span(ast.span)),
            ast::TypeValue::Ptr(pointee) => {
                let pointee = self.type_from_ast(type_namespace, pointee)?;
                let pid = self.get_or_create_pointer_type(Some(pointee));
                Ok(Type::Ptr(pid))
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
            Type::Never | Type::Void => Layout { size: 0, align: 1 },
            Type::Bool => Layout { size: 1, align: 1 },
            Type::Int(i) => Layout {
                size: i.bytes(),
                align: i.bytes(),
            },
            Type::Ptr(_) => Layout {
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

    /// Get a referencse to the pointer description
    pub fn get_ptr(&self, pid: PtrId) -> &Ptr {
        &self.pointers[pid.0]
    }
}

impl Type {
    /// Expect this type to be an integer type, extract the type, and panic otherwise.
    #[track_caller]
    pub fn expect_int(self) -> IntType {
        match self {
            Self::Int(i) => i,
            _ => panic!("Type::expect_int() called on {self:?}"),
        }
    }

    /// Returns `true` if this data type is an integer
    pub fn is_int(self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// Returns `true` if this data type is a signed integer
    pub fn is_signed_int(self) -> bool {
        match self {
            Self::Int(i) => i.is_signed(),
            _ => false,
        }
    }

    /// Combines two types into one, handling the Never type
    ///
    /// 1. If `self` or `other` is Never, the other type is returned.
    /// 2. If both are Never, Never is returned.
    /// 3. Ohterwise require types to be equal, and return the type.
    pub fn comine_ignoring_never(self, other: Self) -> Option<Self> {
        if self == Self::Never {
            Some(other)
        } else if other == Self::Never {
            Some(self)
        } else {
            (self == other).then_some(self)
        }
    }
}
