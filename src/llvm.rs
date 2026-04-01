use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;

use llvm_sys::LLVMIntPredicate;
use llvm_sys::analysis::LLVMVerifyModule;
use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use llvm_sys::target_machine::*;

use crate::common::ArithmeticOp;
use crate::common::CmpOp;
use crate::common::Layout;
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

        let mut ctx = BuildCtx {
            module: &module,
            value_map: HashMap::new(),
        };

        for function in &ir.functions {
            let fn_type = ctx.build_function_type(function);
            let fn_name = function.mangled_name.clone();
            fn_type_map.insert(fn_name.clone(), fn_type);
            fn_map.insert(
                fn_name.clone(),
                module.add_function(&CString::new(fn_name.clone()).unwrap(), fn_type),
            );
        }

        for function in &ir.functions {
            let Some(body) = &function.body else {
                continue;
            };

            let fn_value = &fn_map[&function.mangled_name];
            let postorder = body.postorder();
            ctx.value_map.clear();

            for (i, arg) in body.basic_blokcs[&body.entry].args.iter().enumerate() {
                ctx.value_map.insert(arg.clone(), fn_value.get_param(i));
            }

            let mut basic_blocks_map = HashMap::new();
            for &basic_black_id in postorder.iter().rev() {
                basic_blocks_map.insert(basic_black_id, fn_value.append_basic_block());
            }

            builder.position_at_end(basic_blocks_map[&body.entry]);
            for alloca in &body.allocas {
                ctx.value_map
                    .insert(alloca.definition_id.clone(), builder.alloca(alloca.layout));
            }

            for &basic_block_id in &postorder {
                if basic_block_id != body.entry {
                    builder.position_at_end(basic_blocks_map[&basic_block_id]);
                    for arg in &body.basic_blokcs[&basic_block_id].args {
                        ctx.value_map.insert(arg.clone(), builder.phi(ctx.build_type(arg.ty())));
                    }
                }
            }

            for &basic_block_id in postorder.iter().rev() {
                let basic_block_ir = &body.basic_blokcs[&basic_block_id];
                builder.position_at_end(basic_blocks_map[&basic_block_id]);

                for instruction in &basic_block_ir.instructions {
                    let ty = ctx.build_type(instruction.definition_id.ty());
                    let value = match &instruction.kind {
                        ir::InstructionKind::Load { ptr } => builder.load(ctx.build_value(ptr), ty),
                        ir::InstructionKind::Store { ptr, value } => {
                            builder.store(ctx.build_value(ptr), ctx.build_value(value))
                        }
                        ir::InstructionKind::FunctionCall { name, args } => {
                            let args: Vec<_> = args.iter().map(|arg| ctx.build_value(arg)).collect();
                            builder.function_call(fn_type_map[name], &fn_map[name], &args)
                        }
                        ir::InstructionKind::Cmp { op, signed, lhs, rhs } => builder.cmp(
                            ctx.build_value(lhs),
                            ctx.build_value(rhs),
                            match (op, signed) {
                                (CmpOp::Less, true) => LLVMIntPredicate::LLVMIntSLT,
                                (CmpOp::Less, false) => LLVMIntPredicate::LLVMIntULT,
                                (CmpOp::LessOrEqual, true) => LLVMIntPredicate::LLVMIntSLE,
                                (CmpOp::LessOrEqual, false) => LLVMIntPredicate::LLVMIntULE,
                                (CmpOp::Greater, true) => LLVMIntPredicate::LLVMIntSGT,
                                (CmpOp::Greater, false) => LLVMIntPredicate::LLVMIntUGT,
                                (CmpOp::GreaterOrEqual, true) => LLVMIntPredicate::LLVMIntSGE,
                                (CmpOp::GreaterOrEqual, false) => LLVMIntPredicate::LLVMIntUGE,
                                (CmpOp::Equal, _) => LLVMIntPredicate::LLVMIntEQ,
                                (CmpOp::NotEqual, _) => LLVMIntPredicate::LLVMIntNE,
                            },
                        ),
                        ir::InstructionKind::Arithmetic { op, signed, lhs, rhs } => {
                            let lhs = ctx.build_value(lhs);
                            let rhs = ctx.build_value(rhs);
                            match op {
                                ArithmeticOp::Add => builder.add(lhs, rhs),
                                ArithmeticOp::Sub => builder.sub(lhs, rhs),
                                ArithmeticOp::Mul => builder.mul(lhs, rhs),
                                ArithmeticOp::Div => builder.div(*signed, lhs, rhs),
                            }
                        }
                        ir::InstructionKind::Not { value } => {
                            builder.xor(ctx.build_value(value), ctx.build_value(&ir::Value::Bool(true)))
                        }
                        ir::InstructionKind::OffsetPtr { ptr, offset } => unsafe {
                            let offset = ctx.build_value(offset);
                            LLVMBuildGEP2(
                                builder.raw,
                                LLVMInt8Type(),
                                ctx.build_value(ptr),
                                [offset].as_mut_ptr(),
                                1,
                                c"".as_ptr(),
                            )
                        },
                        ir::InstructionKind::Zext { int } => {
                            builder.zext(ctx.build_value(int), ctx.build_type(instruction.definition_id.ty()))
                        }
                        ir::InstructionKind::Sext { int } => {
                            builder.sext(ctx.build_value(int), ctx.build_type(instruction.definition_id.ty()))
                        }
                        ir::InstructionKind::Truncate { int } => {
                            builder.trunc(ctx.build_value(int), ctx.build_type(instruction.definition_id.ty()))
                        }
                    };
                    ctx.value_map.insert(instruction.definition_id.clone(), value);
                }
                match &basic_block_ir.terminator {
                    ir::Terminator::Jump { to, args } => {
                        builder.jump(basic_blocks_map[to]);
                        for (arg, arg_value) in body.basic_blokcs[to].args.iter().zip(args) {
                            phi_add_incoming(
                                ctx.value_map[arg],
                                basic_blocks_map[&basic_block_id],
                                ctx.build_value(arg_value),
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
                            ctx.build_value(cond),
                            basic_blocks_map[if_true],
                            basic_blocks_map[if_false],
                        );
                        for (arg, arg_value) in body.basic_blokcs[if_true].args.iter().zip(if_true_args) {
                            phi_add_incoming(
                                ctx.value_map[arg],
                                basic_blocks_map[&basic_block_id],
                                ctx.build_value(arg_value),
                            );
                        }
                        for (arg, arg_value) in body.basic_blokcs[if_false].args.iter().zip(if_false_args) {
                            phi_add_incoming(
                                ctx.value_map[arg],
                                basic_blocks_map[&basic_block_id],
                                ctx.build_value(arg_value),
                            );
                        }
                    }
                    ir::Terminator::Return(value) => {
                        builder.ret(ctx.build_value(value));
                    }
                    ir::Terminator::Unreachable => {
                        builder.unreachable();
                    }
                }
            }
        }

        module.verify()?;

        // unsafe {
        //     let target_tripple = LLVMGetDefaultTargetTriple();

        //     let mut target = std::ptr::null_mut();
        //     let mut error_ptr = std::ptr::null_mut();
        //     if LLVMGetTargetFromTriple(target_tripple, &mut target, &mut error_ptr) != 0 {
        //         let error = cstring_from_ptr(error_ptr);
        //         LLVMDisposeMessage(error_ptr);
        //         return Err(error);
        //     }

        //     let target_machine = LLVMCreateTargetMachine(
        //         target,
        //         target_tripple,
        //         c"generic".as_ptr(),
        //         c"".as_ptr(),
        //         LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
        //         LLVMRelocMode::LLVMRelocDefault,
        //         LLVMCodeModel::LLVMCodeModelDefault,
        //     );

        //     let opts = LLVMCreatePassBuilderOptions();
        //     LLVMRunPasses(module.raw, c"default<O1>".as_ptr(), target_machine, opts);
        //     LLVMDisposePassBuilderOptions(opts);
        // }

        Ok(module)
    }

    /// Compile this module for the native target and save the object as a.out.o
    pub fn compile(&self) -> Result<(), CString> {
        unsafe {
            let target_tripple = LLVMGetDefaultTargetTriple();

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
        unsafe {
            LLVM_InitializeAllTargetInfos();
            LLVM_InitializeAllTargets();
            LLVM_InitializeAllTargetMCs();
            LLVM_InitializeAllAsmParsers();
            LLVM_InitializeAllAsmPrinters();
        }

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
    fn alloca(&self, layout: Layout) -> LLVMValueRef {
        unsafe {
            let alloca = LLVMBuildAlloca(self.raw, LLVMArrayType2(LLVMInt8Type(), layout.size), c"".as_ptr());
            LLVMSetAlignment(alloca, layout.align as u32);
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
    fn function_call(&self, fn_ty: LLVMTypeRef, fn_value: &LlvmFunction, args: &[LLVMValueRef]) -> LLVMValueRef {
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

    /// Build the `cmp` instruction
    fn cmp(&self, lhs: LLVMValueRef, rhs: LLVMValueRef, kind: LLVMIntPredicate) -> LLVMValueRef {
        unsafe { LLVMBuildICmp(self.raw, kind, lhs, rhs, c"".as_ptr()) }
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

    /// Build the `div` instruction
    fn div(&self, signed: bool, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
        unsafe {
            match signed {
                false => LLVMBuildUDiv(self.raw, lhs, rhs, c"".as_ptr()),
                true => LLVMBuildSDiv(self.raw, lhs, rhs, c"".as_ptr()),
            }
        }
    }

    /// Build the `xor` instruction
    fn xor(&self, lhs: LLVMValueRef, rhs: LLVMValueRef) -> LLVMValueRef {
        unsafe { LLVMBuildXor(self.raw, lhs, rhs, c"".as_ptr()) }
    }

    /// Build the `trunc` instruction
    fn trunc(&self, int: LLVMValueRef, target_ty: LLVMTypeRef) -> LLVMValueRef {
        unsafe { LLVMBuildTrunc(self.raw, int, target_ty, c"".as_ptr()) }
    }

    /// Build the `zext` instruction
    fn zext(&self, int: LLVMValueRef, target_ty: LLVMTypeRef) -> LLVMValueRef {
        unsafe { LLVMBuildZExt(self.raw, int, target_ty, c"".as_ptr()) }
    }

    /// Build the `sext` instruction
    fn sext(&self, int: LLVMValueRef, target_ty: LLVMTypeRef) -> LLVMValueRef {
        unsafe { LLVMBuildSExt(self.raw, int, target_ty, c"".as_ptr()) }
    }

    /// Build the `br` instruction
    fn jump(&self, to: LLVMBasicBlockRef) -> LLVMValueRef {
        unsafe { LLVMBuildBr(self.raw, to) }
    }

    /// Build the `condbr` instruction
    fn cond_jump(&self, cond: LLVMValueRef, if_true: LLVMBasicBlockRef, if_false: LLVMBasicBlockRef) -> LLVMValueRef {
        unsafe { LLVMBuildCondBr(self.raw, cond, if_true, if_false) }
    }

    /// Build the `ret` instruction
    fn ret(&self, value: LLVMValueRef) -> LLVMValueRef {
        unsafe { LLVMBuildRet(self.raw, value) }
    }

    /// Build the `unreachable` instruction
    fn unreachable(&self) -> LLVMValueRef {
        unsafe { LLVMBuildUnreachable(self.raw) }
    }
}

impl Drop for LlvmBuilder {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.raw) };
    }
}

struct BuildCtx<'a> {
    module: &'a LlvmModule,
    value_map: HashMap<ir::DefinitionId, LLVMValueRef>,
}

impl BuildCtx<'_> {
    /// Construct an LLVM function type for the given declaration
    fn build_function_type(&self, decl: &ir::Function) -> LLVMTypeRef {
        let mut args_ty: Vec<_> = decl.args.iter().map(|ty| self.build_type(ty)).collect();
        unsafe {
            LLVMFunctionType(
                self.build_type(&decl.return_ty),
                args_ty.as_mut_ptr(),
                args_ty.len() as u32,
                decl.is_variadic as i32,
            )
        }
    }

    /// Construct an LLVM type for the given IR type
    fn build_type(&self, ty: &ir::Type) -> LLVMTypeRef {
        unsafe {
            match ty {
                ir::Type::Unit => LLVMStructType([].as_mut_ptr(), 0, 0),
                ir::Type::Bool => LLVMInt1Type(),
                ir::Type::I8 => LLVMInt8Type(),
                ir::Type::I32 => LLVMInt32Type(),
                ir::Type::I64 => LLVMInt64Type(),
                ir::Type::Ptr => LLVMPointerTypeInContext(LLVMGetGlobalContext(), 0),
                ir::Type::Struct(fields) => {
                    let mut fields: Vec<_> = fields.iter().map(|ty| self.build_type(ty)).collect();
                    LLVMStructType(fields.as_mut_ptr(), fields.len() as u32, 0)
                }
                ir::Type::Array(element, length) => LLVMArrayType2(self.build_type(element), *length),
            }
        }
    }

    /// Convert IR value into LLVM value
    fn build_value(&self, value: &ir::Value) -> LLVMValueRef {
        unsafe {
            match value {
                ir::Value::Zst => LLVMConstStruct([].as_mut_ptr(), 0, 0),
                ir::Value::Undefined(ty) => LLVMGetUndef(self.build_type(ty)),
                ir::Value::Bool(bool) => LLVMConstInt(LLVMInt1Type(), *bool as u64, 0),
                ir::Value::String(string) => {
                    let data = CString::new(string.clone()).unwrap();
                    let global = self
                        .module
                        .add_global(LLVMArrayType2(LLVMInt8Type(), (string.len() + 1) as u64));
                    LLVMSetInitializer(global, LLVMConstString(data.as_ptr(), string.len() as u32, 0));
                    global
                }
                ir::Value::Number { data, ty } => LLVMConstInt(LLVMIntType(ty.int_bits().unwrap()), *data as u64, 0),
                ir::Value::Definition(definition_id) => self.value_map[definition_id],
            }
        }
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
