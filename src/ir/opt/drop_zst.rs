//! Drop usages of ZST definitions
//!
//! - Removes ZSTs from function arguments, replaces ZST return type with Unit.
//! - Drop ZST allocas, loads and stores

use super::*;

pub fn pass(function: &mut Function) {
    function.args.retain(|arg| !arg.is_zst());

    if function.return_ty.is_zst() {
        function.return_ty = Type::Unit;
    }

    let Some(body) = &mut function.body else { return };

    body.allocas.retain(|a| a.layout.size > 0);

    for block in body.basic_blokcs.values_mut() {
        block.args.retain(|arg| !arg.ty().is_zst());
        block.instructions.retain_mut(|insn| match &mut insn.kind {
            InstructionKind::Load { ptr: _ } => !insn.definition_id.ty().is_zst(),
            InstructionKind::Store { ptr: _, value } => !value.ty().is_zst(),
            InstructionKind::FunctionCall { name: _, args } => {
                args.retain(|arg| !arg.ty().is_zst());
                true
            }
            InstructionKind::Cmp { .. }
            | InstructionKind::Arithmetic { .. }
            | InstructionKind::Not { .. }
            | InstructionKind::OffsetPtr { .. }
            | InstructionKind::Zext { .. }
            | InstructionKind::Sext { .. }
            | InstructionKind::Truncate { .. } => true,
        });
        match &mut block.terminator {
            Terminator::Jump { args, .. } => {
                args.retain(|arg| !arg.ty().is_zst());
            }
            Terminator::CondJump {
                if_true_args,
                if_false_args,
                ..
            } => {
                if_true_args.retain(|arg| !arg.ty().is_zst());
                if_false_args.retain(|arg| !arg.ty().is_zst());
            }
            Terminator::Return(value) => {
                if value.ty().is_zst() {
                    *value = Value::Zst;
                }
            }
            Terminator::Unreachable => (),
        }
    }
}
