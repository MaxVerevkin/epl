use std::collections::HashMap;
use std::sync::Mutex;

use bumpalo::Bump;

use crate::ast;
use crate::common::{Layout, PtrSize};
use crate::interning::{Interned, Interner};
use crate::ir_tree::{IntType, Struct, StructFieldId, StructInfo, Type, TypeArguments, TypeInfo, TypeParameterOwner};

pub fn run_with_context<R, F>(ptr_size: PtrSize, cb: F) -> R
where
    F: for<'ctx> FnOnce(Context<'ctx>) -> R,
{
    cb(Context {
        inner: &ContextInner::new(&Bump::new(), ptr_size),
    })
}

#[derive(Clone, Copy)]
pub struct Context<'ctx> {
    inner: &'ctx ContextInner<'ctx>,
}

impl<'ctx> Context<'ctx> {
    pub fn ptr_size(self) -> PtrSize {
        self.inner.ptr_size
    }

    pub fn types(self) -> &'ctx Types<'ctx> {
        &self.inner.types
    }

    pub fn get_type_layout_cached(self, ty: Type<'ctx>) -> Option<Layout> {
        self.inner.type_layout_cache.lock().unwrap().get(&ty).copied()
    }

    pub fn cache_type_layout(self, ty: Type<'ctx>, layout: Layout) {
        assert!(
            self.inner
                .type_layout_cache
                .lock()
                .unwrap()
                .insert(ty, layout)
                .is_none()
        );
    }

    /// This must be called exactly once for each struct definition
    pub fn new_struct_unchecked(
        self,
        name: ast::Ident,
        type_parameters: usize,
        fields: Vec<(ast::Ident, StructFieldId)>,
    ) -> Struct<'ctx> {
        Struct(Interned::new_unchecked(self.inner.arena.alloc(StructInfo {
            name,
            type_parameters,
            fields,
        })))
    }

    pub fn register_struct_field_type(self, struct_field_id: StructFieldId, ty: Type<'ctx>, owner: Struct<'ctx>) {
        let prev = self
            .inner
            .type_of_struct_fields
            .lock()
            .unwrap()
            .insert(struct_field_id, (ty, owner));
        assert!(prev.is_none());
    }

    pub fn type_of_struct_field_uninstantiated(self, struct_field_id: StructFieldId) -> Type<'ctx> {
        let (ty, _struct_owner) = *self
            .inner
            .type_of_struct_fields
            .lock()
            .unwrap()
            .get(&struct_field_id)
            .expect("type_of_struct_field called for not-yet-registered struct field");
        ty
    }

    pub fn type_of_struct_field(
        self,
        struct_field_id: StructFieldId,
        type_arguments: TypeArguments<'ctx>,
    ) -> Type<'ctx> {
        let (ty, struct_owner) = *self
            .inner
            .type_of_struct_fields
            .lock()
            .unwrap()
            .get(&struct_field_id)
            .expect("type_of_struct_field called for not-yet-registered struct field");

        ty.instantiate(self, TypeParameterOwner::Struct(struct_owner), type_arguments)
    }

    pub fn offset_of_struct_field(self, struct_field_id: StructFieldId, type_arguments: TypeArguments<'ctx>) -> u64 {
        // TODO: cache this

        let (_ty, struct_owner) = *self
            .inner
            .type_of_struct_fields
            .lock()
            .unwrap()
            .get(&struct_field_id)
            .expect("type_of_struct_field called for not-yet-registered struct field");

        let mut size = 0u64;
        let mut align = 1u64;
        for (_, field_id) in &struct_owner.info().fields {
            let field_ty = self.type_of_struct_field(*field_id, type_arguments);
            let field_layout = field_ty.layout(self);
            size = size.next_multiple_of(field_layout.align);
            align = align.max(field_layout.align);
            if *field_id == struct_field_id {
                return size;
            }
            size += field_layout.size;
        }

        unreachable!()
    }
}

struct ContextInner<'ctx> {
    arena: &'ctx Bump,
    type_info_interner: Interner<'ctx, TypeInfo<'ctx>>,
    type_arguments_interner: Interner<'ctx, [Type<'ctx>]>,
    type_of_struct_fields: Mutex<HashMap<StructFieldId, (Type<'ctx>, Struct<'ctx>)>>,
    types: Types<'ctx>,
    ptr_size: PtrSize,
    type_layout_cache: Mutex<HashMap<Type<'ctx>, Layout>>,
}

impl<'ctx> ContextInner<'ctx> {
    fn new(arena: &'ctx Bump, ptr_size: PtrSize) -> Self {
        let type_info_interner = Interner::new(arena);
        let types = Types::new(&type_info_interner);
        Self {
            arena,
            type_info_interner,
            type_arguments_interner: Interner::new(arena),
            type_of_struct_fields: Mutex::default(),
            types,
            ptr_size,
            type_layout_cache: Mutex::default(),
        }
    }
}

pub struct Types<'ctx> {
    pub never: Type<'ctx>,
    pub unit: Type<'ctx>,
    pub bool: Type<'ctx>,
    pub i8: Type<'ctx>,
    pub u8: Type<'ctx>,
    pub i32: Type<'ctx>,
    pub u32: Type<'ctx>,
    pub i64: Type<'ctx>,
    pub u64: Type<'ctx>,
    pub isize: Type<'ctx>,
    pub usize: Type<'ctx>,
    pub opaque_ptr: Type<'ctx>,
    pub i8_ptr: Type<'ctx>,
}

impl<'ctx> Types<'ctx> {
    fn new(interner: &Interner<'ctx, TypeInfo<'ctx>>) -> Self {
        let i8 = Type::new(interner, TypeInfo::Int(IntType::I8));
        Self {
            never: Type::new(interner, TypeInfo::Never),
            unit: Type::new(interner, TypeInfo::Unit),
            bool: Type::new(interner, TypeInfo::Bool),
            i8,
            u8: Type::new(interner, TypeInfo::Int(IntType::U8)),
            i32: Type::new(interner, TypeInfo::Int(IntType::I32)),
            u32: Type::new(interner, TypeInfo::Int(IntType::U32)),
            i64: Type::new(interner, TypeInfo::Int(IntType::I64)),
            u64: Type::new(interner, TypeInfo::Int(IntType::U64)),
            isize: Type::new(interner, TypeInfo::Int(IntType::ISize)),
            usize: Type::new(interner, TypeInfo::Int(IntType::USize)),
            opaque_ptr: Type::new(interner, TypeInfo::Ptr { pointee: None }),
            i8_ptr: Type::new(interner, TypeInfo::Ptr { pointee: Some(i8) }),
        }
    }
}

impl<'ctx> AsRef<Interner<'ctx, TypeInfo<'ctx>>> for Context<'ctx> {
    fn as_ref(&self) -> &Interner<'ctx, TypeInfo<'ctx>> {
        &self.inner.type_info_interner
    }
}

impl<'ctx> AsRef<Interner<'ctx, [Type<'ctx>]>> for Context<'ctx> {
    fn as_ref(&self) -> &Interner<'ctx, [Type<'ctx>]> {
        &self.inner.type_arguments_interner
    }
}
