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

    /// Returns the `IntType` of this type if it is an integer
    pub fn as_int(self) -> Option<IntType> {
        match self {
            Self::Int(i) => Some(i),
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
                .map(|f| f.offset),
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
            Self::Struct(sid) => typesystem.get_struct(sid).layout,
            Self::Array { element, length } => {
                let mut layout = typesystem.get_type(element).layout(typesystem);
                layout.size = layout.size.next_multiple_of(layout.align);
                layout.size *= length;
                layout
            }
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
    pub layout: Layout,
}

/// A field of a struct definition
#[derive(Debug)]
pub struct StructField {
    pub name: ast::Ident,
    pub ty: Type,
    pub offset: u64,
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

    /// Get a type of a struct definition from its AST representation
    pub fn struct_from_ast(
        &mut self,
        type_namespace: &HashMap<String, Type>,
        ast: &ast::Struct,
        annotations: &BTreeSet<ast::Annotation>,
    ) -> Result<Type, Error> {
        if let Some(annotation) = annotations.iter().next() {
            return Err(
                Error::new(format!("unknown annotation: {:?}", annotation.ident.value)).with_span(annotation.span())
            );
        }

        let mut fields: Vec<StructField> = Vec::new();
        let mut size = 0u64;
        let mut align = 0;
        for field in &ast.fields {
            if fields.iter().any(|x| x.name.value == field.name.value) {
                return Err(Error::new("field with this name already exists").with_span(field.name.span));
            }
            let ty = self.type_from_ast(type_namespace, &field.ty)?;
            let layout = ty.layout(self);
            size = size.next_multiple_of(layout.align);
            align = align.max(layout.align);
            fields.push(StructField {
                name: field.name.clone(),
                ty,
                offset: size,
            });
            size += layout.size;
        }

        let sid = StructId(self.structs.len());
        self.structs.push(Struct {
            name: ast.name.clone(),
            fields,
            layout: Layout {
                size: size.next_multiple_of(align),
                align,
            },
        });
        Ok(Type::Struct(sid))
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
