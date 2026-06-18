use super::*;
use crate::ast;
use crate::common::Layout;
use crate::interning::{Interned, Interner};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Type<'ctx>(pub Interned<'ctx, TypeInfo<'ctx>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeArguments<'ctx>(pub Interned<'ctx, [Type<'ctx>]>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Struct<'ctx>(pub Interned<'ctx, StructInfo>);

impl<'ctx> Type<'ctx> {
    pub fn new(interner: impl AsRef<Interner<'ctx, TypeInfo<'ctx>>>, info: TypeInfo<'ctx>) -> Self {
        Self(interner.as_ref().intern(info))
    }

    pub fn info(self) -> &'ctx TypeInfo<'ctx> {
        self.0.get()
    }

    pub fn layout(self, ctx: Context<'ctx>) -> Layout {
        if let Some(layout) = ctx.get_type_layout_cached(self) {
            return layout;
        }
        let layout = match self.info() {
            TypeInfo::Never | TypeInfo::Unit => Layout { size: 0, align: 1 },
            TypeInfo::Bool => Layout { size: 1, align: 1 },
            TypeInfo::Int(i) => i.layout(ctx),
            TypeInfo::Struct {
                struct_,
                type_arguments,
            } => {
                // TODO: cache this
                let mut size = 0u64;
                let mut align = 1u64;
                for (_, field_id) in &struct_.info().fields {
                    let field_ty = ctx.type_of_struct_field(*field_id, *type_arguments);
                    let field_layout = field_ty.layout(ctx);
                    size = size.next_multiple_of(field_layout.align);
                    align = align.max(field_layout.align);
                    size += field_layout.size;
                }
                size = size.next_multiple_of(align);
                Layout { size, align }
            }
            TypeInfo::Ptr { pointee: _ } => {
                let size = ctx.ptr_size().bytes();
                Layout { size, align: size }
            }
            TypeInfo::Array { element_ty, length } => {
                let element_layout = element_ty.layout(ctx);
                Layout {
                    size: element_layout.size * *length,
                    align: element_layout.align,
                }
            }
            TypeInfo::TypeParameter { .. } => panic!("cannot compute layout of a type parameter"),
        };
        ctx.cache_type_layout(self, layout);
        layout
    }

    pub fn as_struct(self) -> Option<(Struct<'ctx>, TypeArguments<'ctx>)> {
        match self.info() {
            TypeInfo::Struct {
                struct_,
                type_arguments,
            } => Some((*struct_, *type_arguments)),
            _ => None,
        }
    }

    /// Returns `true` if this type is `never`
    pub fn is_never(self) -> bool {
        *self.info() == TypeInfo::Never
    }

    /// Returns `true` if this type is `unit`
    pub fn is_unit(self) -> bool {
        *self.info() == TypeInfo::Unit
    }

    /// Returns `true` if this type is an integer
    pub fn is_int(self) -> bool {
        self.as_int().is_some()
    }

    /// Returns `true` if this type is an integer that is signed
    pub fn is_signed_int(self) -> bool {
        matches!(self.info(), TypeInfo::Int(i) if i.is_signed())
    }

    /// Returns `true` if this data type is a pointer
    pub fn is_ptr(self) -> bool {
        matches!(self.info(), TypeInfo::Ptr { .. })
    }

    /// Returns `true` if this data type is a boolean
    pub fn is_bool(self) -> bool {
        *self.info() == TypeInfo::Bool
    }

    /// Returns the `IntType` of this type if it is an integer
    pub fn as_int(self) -> Option<IntType> {
        match self.info() {
            TypeInfo::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the array element type and array length, if this is an array type
    pub fn as_array(self) -> Option<(Type<'ctx>, u64)> {
        match self.info() {
            TypeInfo::Array {
                element_ty: element,
                length,
            } => Some((*element, *length)),
            _ => None,
        }
    }

    pub fn instantiate(
        self,
        ctx: Context<'ctx>,
        type_parameters_owner: TypeParameterOwner<'ctx>,
        type_arguments: TypeArguments<'ctx>,
    ) -> Self {
        match self.info() {
            TypeInfo::Never | TypeInfo::Unit | TypeInfo::Bool | TypeInfo::Int(_) => self,
            TypeInfo::Struct {
                struct_,
                type_arguments: args,
            } => {
                let instanciated_type_args = args
                    .0
                    .get()
                    .iter()
                    .map(|ty| ty.instantiate(ctx, type_parameters_owner, type_arguments))
                    .collect::<Vec<_>>();
                Type::new(
                    ctx,
                    TypeInfo::Struct {
                        struct_: *struct_,
                        type_arguments: TypeArguments::new(ctx, &instanciated_type_args),
                    },
                )
            }
            TypeInfo::Ptr { pointee } => Type::new(
                ctx,
                TypeInfo::Ptr {
                    pointee: pointee.map(|ty| ty.instantiate(ctx, type_parameters_owner, type_arguments)),
                },
            ),
            TypeInfo::Array { element_ty, length } => Type::new(
                ctx,
                TypeInfo::Array {
                    element_ty: element_ty.instantiate(ctx, type_parameters_owner, type_arguments),
                    length: *length,
                },
            ),
            TypeInfo::TypeParameter { name: _, owner, index } => {
                if *owner == type_parameters_owner {
                    type_arguments.0.get()[*index]
                } else {
                    self
                }
            }
        }
    }

    pub fn render(self) -> String {
        let mut retval = String::new();
        self.render_into(&mut retval);
        retval
    }

    pub fn render_into(self, output: &mut String) {
        use std::fmt::Write;
        match self.info() {
            TypeInfo::Never => output.push('!'),
            TypeInfo::Unit => output.push_str("unit"),
            TypeInfo::Bool => output.push_str("bool"),
            TypeInfo::Int(int_type) => output.push_str(match int_type {
                IntType::I8 => "i8",
                IntType::U8 => "u8",
                IntType::I32 => "i32",
                IntType::U32 => "u32",
                IntType::I64 => "i64",
                IntType::U64 => "u64",
                IntType::ISize => "isize",
                IntType::USize => "usize",
            }),
            TypeInfo::Struct {
                struct_,
                type_arguments,
            } => {
                output.push_str(&struct_.info().name.value);
                if !type_arguments.is_empty() {
                    type_arguments.render_into(output);
                }
            }
            TypeInfo::Ptr { pointee: None } => output.push_str("ptr"),
            TypeInfo::Ptr { pointee: Some(pointee) } => {
                output.push('*');
                pointee.render_into(output);
            }
            TypeInfo::Array { element_ty, length } => {
                output.push('[');
                element_ty.render_into(output);
                write!(output, "; {length}]").unwrap();
            }
            TypeInfo::TypeParameter { name, owner, index: _ } => match owner {
                TypeParameterOwner::Struct(owner) => {
                    write!(output, "`{name} of {}`", owner.info().name.value).unwrap();
                }
                TypeParameterOwner::Function(function_id) => {
                    write!(output, "`{name} of {function_id:?}`").unwrap();
                }
            },
        }
    }
}

impl<'ctx> TypeArguments<'ctx> {
    pub fn new(ctx: Context<'ctx>, args: &[Type<'ctx>]) -> Self {
        let type_args_interner: &Interner<[Type<'ctx>]> = ctx.as_ref();
        Self(type_args_interner.intern_slice(args))
    }

    pub fn is_empty(self) -> bool {
        self.0.get().is_empty()
    }

    pub fn render(self) -> String {
        let mut retval = String::new();
        self.render_into(&mut retval);
        retval
    }

    pub fn render_into(self, output: &mut String) {
        output.push('<');
        for (i, type_arg) in self.0.get().iter().enumerate() {
            type_arg.render_into(output);
            if i + 1 != self.0.get().len() {
                output.push_str(", ");
            }
        }
        output.push('>');
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TypeConstructor<'ctx> {
    NonGeneric(Type<'ctx>),
    Struct(Struct<'ctx>),
}

// TODO: once TypeParameter's name is interned, derive Copy
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeInfo<'ctx> {
    Never,
    Unit,
    Bool,
    Int(IntType),
    Struct { struct_: Struct<'ctx>, type_arguments: TypeArguments<'ctx> },
    Ptr { pointee: Option<Type<'ctx>> },
    Array { element_ty: Type<'ctx>, length: u64 },
    TypeParameter { name: String, owner: TypeParameterOwner<'ctx>, index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeParameterOwner<'ctx> {
    Struct(Struct<'ctx>),
    Function(FunctionId),
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
    ISize,
    USize,
}

impl IntType {
    /// Returns the number of bytes used to store this int
    pub fn bytes(self, ctx: Context) -> u64 {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
            Self::ISize | Self::USize => ctx.ptr_size().bytes(),
        }
    }

    /// Returns `true` if this data type is a signed integer
    pub fn is_signed(self) -> bool {
        match self {
            Self::I8 | Self::I32 | Self::I64 | Self::ISize => true,
            Self::U8 | Self::U32 | Self::U64 | Self::USize => false,
        }
    }

    /// The layout of this integer type
    pub fn layout(self, ctx: Context) -> Layout {
        let size = self.bytes(ctx);
        Layout { size, align: size }
    }
}

impl<'ctx> Struct<'ctx> {
    pub fn info(self) -> &'ctx StructInfo {
        self.0.get()
    }

    pub fn get_field_name(self, field_id: StructFieldId) -> &'ctx ast::Ident {
        &self.info().fields.iter().find(|f| f.1 == field_id).unwrap().0
    }

    pub fn reject_recursive_without_indirection(self, ctx: Context<'ctx>) -> Result<(), Error> {
        #[derive(Debug)]
        struct PathEntry<'ctx> {
            struct_: Struct<'ctx>,
            predecessor: Option<StructFieldId>,
        }

        fn visit<'ctx>(
            ctx: Context<'ctx>,
            path: &mut Vec<PathEntry<'ctx>>,
            this: Struct<'ctx>,
            predecessor: Option<StructFieldId>,
        ) -> Result<(), usize> {
            path.push(PathEntry {
                struct_: this,
                predecessor,
            });

            let i = path.iter().position(|x| x.struct_ == this).unwrap();
            if i + 1 != path.len() {
                return Err(i);
            }

            for (_, field_id) in &this.info().fields {
                visit_ty(
                    ctx,
                    path,
                    ctx.type_of_struct_field_uninstantiated(*field_id),
                    Some(*field_id),
                )?;
            }

            path.pop();

            Ok(())
        }

        fn visit_ty<'ctx>(
            ctx: Context<'ctx>,
            path: &mut Vec<PathEntry<'ctx>>,
            this: Type<'ctx>,
            predecessor: Option<StructFieldId>,
        ) -> Result<(), usize> {
            match this.info() {
                TypeInfo::Never
                | TypeInfo::Unit
                | TypeInfo::Bool
                | TypeInfo::Int(_)
                | TypeInfo::Ptr { .. }
                | TypeInfo::TypeParameter { .. } => (),
                TypeInfo::Struct {
                    struct_,
                    type_arguments,
                } => {
                    visit(ctx, path, *struct_, predecessor)?;
                    for ty in type_arguments.0.get() {
                        visit_ty(ctx, path, *ty, predecessor)?;
                    }
                }
                TypeInfo::Array { element_ty, length: _ } => {
                    visit_ty(ctx, path, *element_ty, predecessor)?;
                }
            }
            Ok(())
        }

        let mut path = Vec::new();
        match visit(ctx, &mut path, self, None) {
            Ok(()) => Ok(()),
            Err(i) => {
                let path = &path[i..];
                let struct_ = path[0].struct_;
                let span = struct_.get_field_name(path[1].predecessor.unwrap()).span;
                let mut msg = format!(
                    "recursive struct definition without indirection (cycle detected): {}",
                    struct_.info().name.value,
                );
                for [before, entry] in path.array_windows() {
                    use std::fmt::Write;
                    let field_name = before.struct_.get_field_name(entry.predecessor.unwrap());
                    write!(msg, ".{} -> {}", field_name.value, entry.struct_.info().name.value).unwrap();
                }
                Err(Error::new(msg).with_span(span))
            }
        }
    }
}

/// A description of a structure
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct StructInfo {
    pub name: ast::Ident,
    pub type_parameters: usize,
    pub fields: Vec<(ast::Ident, StructFieldId)>,
}

/// Parse type from its AST representation
pub fn type_from_ast<'ctx>(
    ctx: Context<'ctx>,
    types_scope: &TypesScope<'ctx>,
    ast: &ast::Type,
) -> Result<Type<'ctx>, Error> {
    Ok(match ast {
        ast::Type::Never(_) => ctx.types().never,
        ast::Type::Ident { ident, type_arguments } => {
            let ty_constructor = types_scope
                .lookup(&ident.value)
                .ok_or_else(|| Error::new(format!("unknown type {:?}", ident.value)).with_span(ident.span))?;
            match ty_constructor {
                TypeConstructor::NonGeneric(ty) => {
                    if let Some(type_arguments) = type_arguments
                        && !type_arguments.arguments.is_empty()
                    {
                        return Err(
                            Error::new(format!("type {:?} is not generic", ident.value)).with_span(type_arguments.span)
                        );
                    }
                    ty
                }
                TypeConstructor::Struct(struct_) => {
                    let type_arguments = type_arguments.as_ref().ok_or_else(|| {
                        Error::new(format!("type {:?} is generic, expected type arguments", ident.value))
                            .with_span(ident.span)
                    })?;
                    if type_arguments.arguments.len() != struct_.info().type_parameters {
                        return Err(Error::new(format!(
                            "type {:?} has {} type parameters, but got {}",
                            ident.value,
                            struct_.info().type_parameters,
                            type_arguments.arguments.len(),
                        ))
                        .with_span(type_arguments.span));
                    }
                    let type_arguments = type_arguments
                        .arguments
                        .iter()
                        .map(|arg| type_from_ast(ctx, types_scope, arg))
                        .collect::<Result<Vec<_>, _>>()?;
                    Type::new(
                        ctx,
                        TypeInfo::Struct {
                            struct_,
                            type_arguments: TypeArguments::new(ctx, &type_arguments),
                        },
                    )
                }
            }
        }
        ast::Type::Ptr { star_span: _, pointee } => {
            let pointee = type_from_ast(ctx, types_scope, pointee)?;
            Type::new(ctx, TypeInfo::Ptr { pointee: Some(pointee) })
        }
        ast::Type::Array {
            element_type,
            length,
            span: _,
        } => {
            let element_ty = type_from_ast(ctx, types_scope, element_type)?;
            let length = match &length.kind {
                ast::ExprKind::Literal(ast::Literal::Number(num)) => *num as u64,
                _ => return Err(Error::new("array length must be a number literal").with_span(length.span)),
            };
            Type::new(ctx, TypeInfo::Array { element_ty, length })
        }
    })
}

#[derive(Default)]
pub struct TypesScope<'ctx> {
    pub by_name: HashMap<String, TypeConstructor<'ctx>>,
    parent: Option<Box<Self>>,
}

impl<'ctx> TypesScope<'ctx> {
    /// Create a nested scope
    pub fn push(&mut self) {
        let parent = std::mem::take(self);
        self.parent = Some(Box::new(parent));
    }

    /// Pop the latest nested scope
    pub fn pop(&mut self) {
        *self = *self.parent.take().unwrap();
    }

    /// Lookup a type by its name, recursively traversing the list of scopes
    pub fn lookup(&self, name: &str) -> Option<TypeConstructor<'ctx>> {
        if let Some(ty) = self.by_name.get(name) {
            return Some(*ty);
        }
        if let Some(parent) = &self.parent {
            return parent.lookup(name);
        }
        None
    }

    pub fn insert_ast_type_parameters(
        &mut self,
        ctx: Context<'ctx>,
        owner: TypeParameterOwner<'ctx>,
        type_parameters: &[ast::TypeParameter],
    ) -> Result<(), Error> {
        for (type_parameter_i, type_parameter) in type_parameters.iter().enumerate() {
            if self
                .by_name
                .insert(
                    type_parameter.name.value.clone(),
                    TypeConstructor::NonGeneric(Type::new(
                        ctx,
                        TypeInfo::TypeParameter {
                            name: type_parameter.name.value.clone(),
                            owner,
                            index: type_parameter_i,
                        },
                    )),
                )
                .is_some()
            {
                return Err(
                    Error::new("type parameter with this name already exists").with_span(type_parameter.name.span)
                );
            }
        }
        Ok(())
    }
}
