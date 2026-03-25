use std::fmt::Write;

use super::*;

pub fn dump(module: &Module) -> String {
    let mut output = String::new();

    for function in module.functions.values() {
        dump_function_desc(&mut output, function, module);
        match &function.body {
            Some(body) => {
                output.push(' ');
                dump_block_expr(&mut output, body, module, 0);
            }
            None => output.push_str(";\n"),
        }
        output.push('\n');
    }

    output
}

fn dump_function_desc(output: &mut String, function: &Function, module: &Module) {
    output.push_str("fn ");
    output.push_str(&function.name.value);
    output.push('(');
    for (arg_i, (arg_name, arg_ty)) in function.args.iter().enumerate() {
        output.push_str(arg_name);
        output.push_str(": ");
        dump_type(output, *arg_ty, module);
        if arg_i + 1 != function.args.len() || function.is_variadic {
            output.push_str(", ");
        }
    }
    if function.is_variadic {
        output.push_str("...");
    }
    output.push_str(") -> ");
    dump_type(output, function.return_ty, module);
}

fn dump_type(output: &mut String, ty: Type, module: &Module) {
    match ty {
        Type::Never => output.push('!'),
        Type::Unit => output.push_str("unit"),
        Type::Bool => output.push_str("bool"),
        Type::Int(int_type) => output.push_str(match int_type {
            IntType::I8 => "i8",
            IntType::U8 => "u8",
            IntType::I32 => "i32",
            IntType::U32 => "u32",
            IntType::I64 => "i64",
            IntType::U64 => "u64",
        }),
        Type::Struct(struct_id) => write!(output, "{struct_id:?}").unwrap(),
        Type::Ptr { pointee: None } => output.push_str("ptr"),
        Type::Ptr { pointee: Some(pointee) } => {
            output.push('*');
            dump_type(output, module.typesystem.get_type(pointee), module);
        }
        Type::Array { element, length } => {
            output.push('[');
            dump_type(output, module.typesystem.get_type(element), module);
            write!(output, "; {length}]").unwrap();
        }
    }
}

fn dump_block_expr(output: &mut String, body: &BlockExpr, module: &Module, indent_level: u32) {
    output.push_str("{\n");
    for (var, ty) in &body.variables {
        indent(output, indent_level + 1);
        write!(output, "{var:?}: ").unwrap();
        dump_type(output, *ty, module);
        output.push('\n');
    }
    for expr in &body.exprs {
        dump_expr(output, expr.as_ref(), module, indent_level + 1);
    }
    indent(output, indent_level);
    output.push_str("}\n");
}

fn dump_expr(output: &mut String, expr: ExprRef, module: &Module, indent_level: u32) {
    indent(output, indent_level);
    match expr {
        ExprRef::R(expr) => {
            dump_rexpr(output, expr, module);
            output.push_str(" TYPE=");
            dump_type(output, expr.ty, module);
        }
        ExprRef::L(expr) => {
            dump_lexpr(output, expr, module);
            output.push_str(" TYPE=");
            dump_type(output, expr.ty, module);
        }
    }
    output.push('\n');
    expr.visit_children(|expr| dump_expr(output, expr, module, indent_level + 1));
}

fn dump_rexpr(output: &mut String, expr: &RExpr, module: &Module) {
    match &expr.kind {
        RExprKind::Undefined => output.push_str("UNDEFINED"),
        RExprKind::ConstUnit => output.push_str("UNIT"),
        RExprKind::ConstNumber(num) => write!(output, "CONST_NUMBER({num})").unwrap(),
        RExprKind::ConstString(str) => write!(output, "CONST_STRING({str:?})").unwrap(),
        RExprKind::ConstBool(bool) => write!(output, "CONST_BOOL({bool})").unwrap(),
        RExprKind::Field(_, field) => write!(output, "FIELD_ACCESS({field})").unwrap(),
        RExprKind::ArrayElement(_, _) => output.push_str("ARRAY_ELEMENT"),
        RExprKind::Store(_, _) => output.push_str("STORE"),
        RExprKind::GetPointer(_) => output.push_str("GET_POINTER"),
        RExprKind::Block(_) => output.push_str("BLOCK"), // TODO: enumerate variables
        RExprKind::Return(_) => output.push_str("RETURN"),
        RExprKind::Break(loop_id, _) => write!(output, "BREAK({loop_id:?})").unwrap(),
        RExprKind::BinOp(op, _, _) => write!(output, "BIN_OP({op:?})").unwrap(),
        RExprKind::If { .. } => output.push_str("IF"),
        RExprKind::Loop(loop_id, _) => write!(output, "LOOP({loop_id:?})").unwrap(),
        RExprKind::ArrayInitializer(_) => output.push_str("ARRAY_INITIALIZER"),
        RExprKind::StructInitializer(_) => output.push_str("STRUCT_INITIALIZER"),
        RExprKind::FunctionCall(function_id, _) => write!(
            output,
            "FUNCTION_CALL({:?})",
            module.functions.get(function_id).unwrap().name.value
        )
        .unwrap(),
        RExprKind::Cast(_) => output.push_str("CAST"),
        RExprKind::Not(_) => output.push_str("NOT"),
    }
}

fn dump_lexpr(output: &mut String, expr: &LExpr, module: &Module) {
    match &expr.kind {
        LExprKind::Dereference(_) => output.push_str("DEREFERENCE"),
        LExprKind::Variable(variable_id) => write!(output, "VARIABLE({variable_id:?})").unwrap(),
        LExprKind::Field(_, field) => write!(output, "FIELD({field:?})").unwrap(),
        LExprKind::ArrayElement(_, _) => output.push_str("ARRAY_ELEMENT"),
    }
}

fn indent(output: &mut String, indent: u32) {
    if indent > 0 {
        output.push(' ');
        output.push(' ');
        output.push(' ');
        output.push(' ');
        for _ in 1..indent {
            output.push('|');
            output.push(' ');
            output.push(' ');
            output.push(' ');
        }
    }
}
