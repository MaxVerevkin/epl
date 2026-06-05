use std::num::NonZeroUsize;

use super::*;
use crate::ast;
use crate::common::Layout;

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

    /// Returns `true` if this data type is an integer
    pub fn is_int(self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// Returns `true` if this data type is an pointer
    pub fn is_ptr(self) -> bool {
        matches!(self, Self::Ptr { .. })
    }

    /// Returns the `IntType` of this type if it is an integer
    pub fn as_int(self) -> Option<IntType> {
        match self {
            Self::Int(i) => Some(i),
            _ => None,
        }
    }

    /// Returns the `StructId` of this type if it is a struct
    pub fn as_struct(self) -> Option<StructId> {
        match self {
            Self::Struct(s) => Some(s),
            _ => None,
        }
    }

    /// Returns `true` if this data type is an integer that is signed
    pub fn is_signed_int(self) -> bool {
        matches!(self, Self::Int(i) if i.is_signed())
    }

    /// Returns the type ID of the array's element, or None if not an array
    pub fn array_element_type_id(self) -> Option<TypeId> {
        match self {
            Self::Array { element, length: _ } => Some(element),
            _ => None,
        }
    }

    /// Returns the type of the array's element, or None if not an array
    pub fn array_element_type(self, typesystem: &TypeSystem) -> Option<Self> {
        self.array_element_type_id().map(|id| typesystem.get_type(id))
    }

    /// Returns the byte offset of the struct's field
    pub fn get_field_offset(self, name: &str, typesystem: &TypeSystem) -> Option<u64> {
        match self {
            Self::Struct(struct_id) => typesystem
                .get_struct(struct_id)
                .fields
                .iter()
                .find(|f| f.name.value == name)
                .map(|f| f.offset.unwrap()),
            _ => None,
        }
    }

    /// Get physical layout of this type
    pub fn layout(self, typesystem: &TypeSystem) -> Layout {
        match self {
            Self::Never | Self::Unit => Layout { size: 0, align: 1 },
            Self::Bool => Layout { size: 1, align: 1 },
            Self::Int(i) => Layout {
                size: i.bytes(),
                align: i.bytes(),
            },
            Self::Ptr { pointee: _ } => Layout {
                size: typesystem.ptr_size,
                align: typesystem.ptr_size,
            },
            Self::Struct(sid) => typesystem.get_struct(sid).layout.unwrap(),
            Self::Array { element, length } => {
                let mut layout = typesystem.get_type(element).layout(typesystem);
                layout.size = layout.size.next_multiple_of(layout.align);
                layout.size *= length;
                layout
            }
        }
    }
}

/// Integer data type
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
pub struct StructId(NonZeroUsize);

/// A description of a structure
#[derive(Debug)]
pub struct Struct {
    pub name: ast::Ident,
    pub fields: Vec<StructField>,
    pub layout: Option<Layout>, // None during IR_TREE construction
}

/// A field of a struct definition
#[derive(Debug)]
pub struct StructField {
    pub name: ast::Ident,
    pub ty: Type,
    pub offset: Option<u64>, // None during IR_TREE construction
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
    structs: Vec<Option<Struct>>,
    types_with_ids: Vec<Type>,
    type_lut: HashMap<Type, TypeId>,
}

impl TypeSystem {
    /// Create a new type system context. Generally there should be only one context created.
    pub fn new(ptr_size: u64) -> Self {
        Self {
            ptr_size,
            structs: vec![None], // Dummy zero'th struct
            types_with_ids: vec![Type::Int(IntType::I8)],
            type_lut: [(Type::Int(IntType::I8), TypeId::I8)].into_iter().collect(),
        }
    }

    /// Allocate the next struct ID. The struct must be then defined via `define_struct` for the ID to be usable.
    pub fn alloc_struct_id(&mut self) -> StructId {
        let id = StructId(NonZeroUsize::new(self.structs.len()).unwrap());
        self.structs.push(None);
        id
    }

    /// Define the structure
    ///
    /// # Panics
    ///
    /// Panics if called more than once for the same ID.
    pub fn define_struct(&mut self, id: StructId, def: Struct) {
        if self.structs[id.0.get()].is_some() {
            panic!("define_struct() called twice with the same ID");
        }

        self.structs[id.0.get()] = Some(def);
    }

    /// Resolve the layout of a struct, recursively resolving layouts as needed.
    pub fn resolve_layout(&mut self, ty: Type, path_stack: &mut Vec<Type>) -> Result<(), Error> {
        path_stack.push(ty);

        if let Some(i) = path_stack.iter().position(|x| *x == ty)
            && i + 1 != path_stack.len()
        {
            use std::fmt::Write;
            let mut msg = String::from("could not resolve type layout, dependency cycle detected: ");
            for (entry_i, entry) in path_stack[i..].iter().enumerate() {
                if entry_i != 0 {
                    msg.push_str(" -> ");
                }
                write!(msg, "{entry:?}").unwrap();
            }
            let span = match path_stack.first().unwrap() {
                Type::Struct(struct_id) => self.get_struct(*struct_id).name.span,
                Type::Never | Type::Unit | Type::Bool | Type::Int(_) | Type::Ptr { .. } | Type::Array { .. } => {
                    unreachable!()
                }
            };
            return Err(Error::new(msg).with_span(span));
        }

        match ty {
            Type::Struct(struct_id) => {
                let field_layouts = self
                    .get_struct(struct_id)
                    .fields
                    .iter()
                    .map(|f| f.ty)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|ty| {
                        self.resolve_layout(ty, path_stack)?;
                        Ok(ty.layout(self))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let s = self.structs[struct_id.0.get()].as_mut().unwrap();
                let mut size = 0u64;
                let mut align = 1;
                for (field, layout) in s.fields.iter_mut().zip(field_layouts) {
                    size = size.next_multiple_of(layout.align);
                    align = align.max(layout.align);
                    field.offset = Some(size);
                    size += layout.size;
                }
                s.layout = Some(Layout { size, align });
            }
            Type::Array { element, length: _ } => {
                self.resolve_layout(self.get_type(element), path_stack)?;
            }
            Type::Never | Type::Unit | Type::Bool | Type::Int(_) | Type::Ptr { .. } => (),
        }

        Ok(())
    }

    /// Returns the target pointer size
    pub fn ptr_size(&self) -> u64 {
        self.ptr_size
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
            ast::Type::Ptr { star_span: _, pointee } => Ok(self.type_from_ast(type_namespace, pointee)?.make_ptr(self)),
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
                        value: ast::LiteralExprValue::Number(num, _),
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

    /// Get a reference to the struct declaration
    ///
    /// # Panics
    ///
    /// Panics if called with an ID that was not yet defined via `define_struct`.
    pub fn get_struct(&self, sid: StructId) -> &Struct {
        match &self.structs[sid.0.get()] {
            Some(def) => def,
            None => panic!("get_struct called with ID that was not defined"),
        }
    }

    /// Get the actual type by ID
    pub fn get_type(&self, id: TypeId) -> Type {
        self.types_with_ids[id.0]
    }
}
