use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;

use llvm_sys::analysis::LLVMVerifyModule;
use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use llvm_sys::target_machine::*;

use crate::ir;

/// An LLVM module
pub struct LlvmModule {
    raw: LLVMModuleRef,
}

impl LlvmModule {
    /// Construct an LLVM module from IR
    pub fn from_ir(ir: &ir::Ir) -> Result<Self, CString> {
        let module = Self::new(c"epl_module");
        let builder = LlvmBuilder::new();
        let mut fn_type_map = HashMap::new();
        let mut fn_map = HashMap::new();

        for (fn_name, fn_decl) in &ir.function_decls {
            let fn_type = build_function_type(fn_decl);
            fn_type_map.insert(fn_name.clone(), fn_type);
            fn_map.insert(
                fn_name.clone(),
                module.add_function(&CString::new(fn_name.clone()).unwrap(), fn_type),
            );
        }

        for (fn_name, fn_ir) in &ir.functions {
            let fn_value = &fn_map[fn_name];
            let mut value_map = HashMap::new();
            let postorder = fn_ir.postorder();

            for (i, &arg) in fn_ir.basic_blokcs[&fn_ir.entry].args.iter().enumerate() {
                value_map.insert(arg, fn_value.get_param(i));
            }

            let mut basic_blocks_map = HashMap::new();
            for &basic_black_id in postorder.iter().rev() {
                basic_blocks_map.insert(basic_black_id, fn_value.append_basic_block());
            }

            builder.position_at_end(basic_blocks_map[&fn_ir.entry]);
            for alloca in &fn_ir.allocas {
                value_map.insert(
                    alloca.definition_id,
                    builder.alloca(alloca.size, alloca.align),
                );
            }

            for &basic_block_id in fn_ir.basic_blokcs.keys() {
                if basic_block_id != fn_ir.entry {
                    builder.position_at_end(basic_blocks_map[&basic_block_id]);
                    for &arg in &fn_ir.basic_blokcs[&basic_block_id].args {
                        value_map.insert(arg, builder.phi(build_type(arg.ty())));
                    }
                }
            }

            for &basic_block_id in postorder.iter().rev() {
                let basic_block_ir = &fn_ir.basic_blokcs[&basic_block_id];
                builder.position_at_end(basic_blocks_map[&basic_block_id]);

                for instruction in &basic_block_ir.instructions {
                    let ty = build_type(instruction.definition_id.ty());
                    let value = match &instruction.kind {
                        ir::InstructionKind::Load { ptr } => {
                            builder.load(build_value(ptr, &value_map, &module), ty)
                        }
                        ir::InstructionKind::Store { ptr, value } => builder.store(
                            build_value(ptr, &value_map, &module),
                            build_value(value, &value_map, &module),
                        ),
                        ir::InstructionKind::FunctionCall { name, args } => {
                            let args: Vec<_> = args
                                .iter()
                                .map(|arg| build_value(arg, &value_map, &module))
                                .collect();
                            builder.function_call(fn_type_map[name], &fn_map[name], &args)
                        }
                        ir::InstructionKind::CmpSL { lhs, rhs } => builder.cmp_sl(
                            build_value(lhs, &value_map, &module),
                            build_value(rhs, &value_map, &module),
                        ),
                        ir::InstructionKind::CmpUL { lhs, rhs } => builder.cmp_ul(
                            build_value(lhs, &value_map, &module),
                            build_value(rhs, &value_map, &module),
                        ),
                        ir::InstructionKind::Add { lhs, rhs } => builder.add(
                            build_value(lhs, &value_map, &module),
                            build_value(rhs, &value_map, &module),
                        ),
                        ir::InstructionKind::Sub { lhs, rhs } => builder.sub(
                            build_value(lhs, &value_map, &module),
                            build_value(rhs, &value_map, &module),
                        ),
                        ir::InstructionKind::Mul { lhs, rhs } => builder.mul(
                            build_value(lhs, &value_map, &module),
                            build_value(rhs, &value_map, &module),
                        ),
                    };
                    value_map.insert(instruction.definition_id, value);
                }
                match &basic_block_ir.terminator {
                    ir::Terminator::Jump { to, args } => {
                        builder.jump(basic_blocks_map[to]);
                        for (&arg, arg_value) in fn_ir.basic_blokcs[to].args.iter().zip(args) {
                            phi_add_incoming(
                                value_map[&arg],
                                basic_blocks_map[&basic_block_id],
                                build_value(arg_value, &value_map, &module),
                            );
                        }
                    }
                    ir::Terminator::CondJump {
                        cond,
                        if_true,
                        if_true_args,
                        if_false,
                        if_false_args,
                    } => {
                        builder.cond_jump(
                            build_value(cond, &value_map, &module),
                            basic_blocks_map[if_true],
                            basic_blocks_map[if_false],
                        );
                        for (&arg, arg_value) in
                            fn_ir.basic_blokcs[if_true].args.iter().zip(if_true_args)
                        {
                            phi_add_incoming(
                                value_map[&arg],
                                basic_blocks_map[&basic_block_id],
                                build_value(arg_value, &value_map, &module),
                            );
                        }
                        for (&arg, arg_value) in
                            fn_ir.basic_blokcs[if_false].args.iter().zip(if_false_args)
                        {
                            phi_add_incoming(
                                value_map[&arg],
                                basic_blocks_map[&basic_block_id],
                                build_value(arg_value, &value_map, &module),
                            );
                        }
                    }
                    ir::Terminator::Return { value } => {
                        builder.ret(build_value(value, &value_map, &module));
                    }
                    ir::Terminator::Unreachable => todo!(),
                }
            }
        }

        module.verify()?;
        Ok(module)
    }

    /// Compile this module for the native target and save the object as a.out.o
    pub fn compile(&self) -> Result<(), CString> {
        unsafe {
            let target_tripple = LLVMGetDefaultTargetTriple();

            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllAsmParsers();
            LLVM_InitializeAllAsmPrinters();

            let mut target = std::ptr::null_mut();
            let mut error_ptr = std::ptr::null_mut();
            if LLVMGetTargetFromTriple(target_tripple, &mut target, &mut error_ptr) != 0 {
                let error = cstring_from_ptr(error_ptr);
                LLVMDisposeMessage(error_ptr);
                return Err(error);
            }

            let target_machine = LLVMCreateTargetMachine(
                target,
                target_tripple,
                c"generic".as_ptr(),
                c"".as_ptr(),
                LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
                LLVMRelocMode::LLVMRelocDefault,
                LLVMCodeModel::LLVMCodeModelDefault,
            );

            LLVMSetModuleDataLayout(self.raw, LLVMCreateTargetDataLayout(target_machine));
            LLVMSetTarget(self.raw, target_tripple);

            if LLVMTargetMachineEmitToFile(
                target_machine,
                self.raw,
                c"a.out.o".as_ptr(),
                LLVMCodeGenFileType::LLVMObjectFile,
                &mut error_ptr,
            ) != 0
            {
                let error = cstring_from_ptr(error_ptr);
                LLVMDisposeMessage(error_ptr);
                return Err(error);
            }

            Ok(())
        }
    }

    /// Create a new module with a given name
    fn new(name: &CStr) -> Self {
        Self {
            raw: unsafe { LLVMModuleCreateWithName(name.as_ptr()) },
        }
    }

    /// Add a function to this module
    fn add_function(&self, name: &CStr, ty: LLVMTypeRef) -> LlvmFunction {
        let raw = unsafe { LLVMAddFunction(self.raw, name.as_ptr(), ty) };
        LlvmFunction { raw }
    }

    /// Verify the module
    fn verify(&self) -> Result<(), CString> {
        let mut error_ptr = std::ptr::null_mut();
        let status = unsafe {
            LLVMVerifyModule(
                self.raw,
                llvm_sys::analysis::LLVMVerifierFailureAction::LLVMReturnStatusAction,
                &mut error_ptr,
            )
        };
        if status == 1 {
            let error = unsafe { cstring_from_ptr(error_ptr) };
            unsafe { LLVMDisposeMessage(error_ptr) };
            Err(error)
        } else {
            unsafe { LLVMDisposeMessage(error_ptr) };
            Ok(())
        }
    }

    /// Add global to the module
    fn add_global(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        unsafe { LLVMAddGlobal(self.raw, ty, c"".as_ptr()) }
    }
}

impl Drop for LlvmModule {
    fn drop(&mut self) {
        unsafe { LLVMDisposeModule(self.raw) };
    }
}

impl fmt::Display for LlvmModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            let s_ptr = LLVMPrintModuleToString(self.raw);
            let s = CStr::from_ptr(s_ptr);
            let s = s.to_str().unwrap();
            f.write_str(s)?;
            LLVMDisposeMessage(s_ptr);
            Ok(())
        }
    }
}

struct LlvmFunction {
    raw: LLVMValueRef,
}

impl LlvmFunction {
    /// Append a new basic block to this function
    fn append_basic_block(&self) -> LLVMBasicBlockRef {
        unsafe { LLVMAppendBasicBlock(self.raw, c"".as_ptr()) }
    }

    /// Get the value of the ith parameter
    fn get_param(&self, i: usize) -> LLVMValueRef {
        unsafe { LLVMGetParam(self.raw, i as u32) }
    }
}

struct LlvmBuilder {
    raw: LLVMBuilderRef,
}

impl LlvmBuilder {
    /// Create a new instruction builder
    fn new() -> Self {
        Self {
            raw: unsafe { LLVMCreateBuilder() },
        }
    }

    /// Position this builder at the end of a given basic block
    fn position_at_end(&self, of: LLVMBasicBlockRef) {
        unsafe { LLVMPositionBuilderAtEnd(self.raw, of) };
    }

    /// Build the `alloca` instruction
    fn alloca(&self, size: u64, align: u64) -> LLVMValueRef {
        unsafe {
            let alloca =
                LLVMBuildAlloca(self.raw, LLVMArrayType2(LLVMInt8Type(), size), c"".as_ptr());
            LLVMSetAlignment(alloca, align as u32);
            alloca
        }
    }

    /// Build the `phi` instruction
    fn phi(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        unsafe { LLVMBuildPhi(self.raw, ty, c"".as_ptr()) }
    }

    /// Build the `load` instruction
    fn load(&self, ptr: LLVMValueRef, ty: LLVMTypeRef) -> LLVMValueRef {
        unsafe { LLVMBuildLoad2(self.raw, ty, ptr, c"".as_ptr()) }
    }

    /// Build the `store` instruction
    fn store(&self, ptr: LLVMValueRef, value: LLVMValueRef) -> LLVMValueRef {
        unsafe { LLVMBuildStore(self.raw, value, ptr) }
    }

    /// Build the `call` instruction
    fn function_call(
        &self,
        fn_ty: LLVMTypeRef,
        fn_value: &LlvmFunction,
        args: &[LLVMValueRef],
    ) -> LLVMValueRef {
        unsafe {
            LLVMBuildCall2(
                self.raw,
                fn_ty,
                fn_value.raw,
                args.as_ptr().cast_mut(),
                args.len() as u32,
                c"".as_ptr(),
            )
        }
    }

    /// Build the `cmp` "signed less than" instruction
    fn cmp_sl(&self, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
        unsafe {
            LLVMBuildICmp(
                self.raw,
                llvm_sys::LLVMIntPredicate::LLVMIntSLT,
                lhs,
                rhs,
                c"".as_ptr(),
            )
        }
    }

    /// Build the `cmp` "unsigned less than" instruction
    fn cmp_ul(&self, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
        unsafe {
            LLVMBuildICmp(
                self.raw,
                llvm_sys::LLVMIntPredicate::LLVMIntULT,
                lhs,
                rhs,
                c"".as_ptr(),
            )
        }
    }

    /// Build the `add` instruction
    fn add(&self, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
        unsafe { LLVMBuildAdd(self.raw, lhs, rhs, c"".as_ptr()) }
    }

    /// Build the `sub` instruction
    fn sub(&self, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
        unsafe { LLVMBuildSub(self.raw, lhs, rhs, c"".as_ptr()) }
    }

    /// Build the `mul` instruction
    fn mul(&self, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
        unsafe { LLVMBuildMul(self.raw, lhs, rhs, c"".as_ptr()) }
    }

    /// Build the `br` instruction
    fn jump(&self, to: LLVMBasicBlockRef) -> LLVMValueRef {
        unsafe { LLVMBuildBr(self.raw, to) }
    }

    /// Build the `condbr` instruction
    fn cond_jump(
        &self,
        cond: LLVMValueRef,
        if_true: LLVMBasicBlockRef,
        if_false: LLVMBasicBlockRef,
    ) -> LLVMValueRef {
        unsafe { LLVMBuildCondBr(self.raw, cond, if_true, if_false) }
    }

    /// Build the `ret` instruction
    fn ret(&self, value: LLVMValueRef) -> LLVMValueRef {
        unsafe { LLVMBuildRet(self.raw, value) }
    }
}

impl Drop for LlvmBuilder {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.raw) };
    }
}

/// Construct an LLVM function type for the given declaration
fn build_function_type(decl: &ir::FunctionDecl) -> LLVMTypeRef {
    let mut args_ty: Vec<_> = decl.args.iter().map(|arg| build_type(arg.ty)).collect();
    unsafe {
        LLVMFunctionType(
            build_type(decl.return_ty),
            args_ty.as_mut_ptr(),
            args_ty.len() as u32,
            0,
        )
    }
}

/// Construct an LLVM type for the given IR type
fn build_type(ty: ir::Type) -> LLVMTypeRef {
    unsafe {
        match ty {
            ir::Type::Never | ir::Type::Void => LLVMStructType([].as_mut_ptr(), 0, 0),
            ir::Type::Bool => LLVMInt1Type(),
            ir::Type::I32 | ir::Type::U32 => LLVMInt32Type(),
            ir::Type::CStr | ir::Type::OpaquePointer => {
                LLVMPointerTypeInContext(LLVMGetGlobalContext(), 0)
            }
        }
    }
}

/// Convert IR value into LLVM value
fn build_value(
    value: &ir::Value,
    value_map: &HashMap<ir::DefinitionId, LLVMValueRef>,
    module: &LlvmModule,
) -> LLVMValueRef {
    match value {
        ir::Value::Definition(definition_id) => value_map[definition_id],
        ir::Value::Constant(constant) => unsafe {
            match constant {
                ir::Constant::Void => LLVMConstStruct([].as_mut_ptr(), 0, 0),
                ir::Constant::Bool(bool) => LLVMConstInt(LLVMInt1Type(), *bool as u64, 0),
                ir::Constant::String(string) => {
                    let data = CString::new(string.clone()).unwrap();
                    let global = module
                        .add_global(LLVMArrayType2(LLVMInt8Type(), (string.len() + 1) as u64));
                    LLVMSetInitializer(
                        global,
                        LLVMConstString(data.as_ptr(), string.len() as u32, 0),
                    );
                    global
                }
                ir::Constant::Number { data, bits, signed } => {
                    LLVMConstInt(LLVMIntType(*bits as u32), *data as u64, *signed as i32)
                }
            }
        },
    }
}

fn phi_add_incoming(phi: LLVMValueRef, from_block: LLVMBasicBlockRef, value: LLVMValueRef) {
    unsafe {
        LLVMAddIncoming(phi, [value].as_mut_ptr(), [from_block].as_mut_ptr(), 1);
    }
}

unsafe fn cstring_from_ptr(ptr: *const std::ffi::c_char) -> CString {
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_owned()
}
