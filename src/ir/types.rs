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
    I32,
    U32,
    CStr,
    OpaquePointer,
    Struct(StructId),
    Ptr(PtrId),
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

/// A description of a pointer
#[derive(Debug)]
pub struct Ptr {
    pub pointee: Type,
}

/// Store the state of the type system
#[derive(Debug)]
pub struct TypeSystem {
    ptr_size: u64,
    structs: Vec<Struct>,
    pointers: Vec<Ptr>,
    pointer_lut: HashMap<Type, PtrId>, // pointee -> poniter id
}

impl TypeSystem {
    /// Create a new type system context. Generally there should be only one context created.
    pub fn new(ptr_size: u64) -> Self {
        Self {
            ptr_size,
            structs: Vec::new(),
            pointers: Vec::new(),
            pointer_lut: HashMap::new(),
        }
    }

    /// Get a pointer ID given the pointee, or allocate and cache a new one
    pub fn get_or_create_pointer_type(&mut self, pointee: Type) -> PtrId {
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
                let pid = self.get_or_create_pointer_type(pointee);
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
            Type::I32 | Type::U32 => Layout { size: 4, align: 4 },
            Type::CStr | Type::OpaquePointer | Type::Ptr(_) => Layout {
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
}

impl Type {
    /// Returns `true` if this data type is an integer
    pub fn is_int(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Returns `true` if this data type is a signed integer
    pub fn is_signed_int(self) -> bool {
        matches!(self, Self::I32)
    }

    /// Returns `true` if this data type is an unsigned integer
    pub fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U32)
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
