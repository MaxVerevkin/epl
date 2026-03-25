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
        indent(output, indent_level + 1);
        dump_expr(output, expr, module, indent_level + 1);
    }
    indent(output, indent_level);
    output.push_str("}\n");
}

fn dump_expr(output: &mut String, expr: &Expr, module: &Module, indent_level: u32) {
    match expr {
        Expr::R(expr) => dump_rexpr(output, expr, module, indent_level),
        Expr::L(expr) => dump_lexpr(output, expr, module, indent_level),
    }
}

fn dump_rexpr(output: &mut String, expr: &RExpr, module: &Module, indent_level: u32) {
    match &expr.kind {
        RExprKind::Undefined => output.push_str("undefined\n"),
        RExprKind::ConstUnit => output.push_str("unit\n"),
        RExprKind::ConstNumber(num) => write!(output, "{num}\n").unwrap(),
        RExprKind::ConstString(str) => write!(output, "{str:?}\n").unwrap(),
        RExprKind::ConstBool(bool) => write!(output, "{bool}\n").unwrap(),
        RExprKind::Field(expr, field) => {
            output.push_str("FIELD_ACCESS (");
            output.push_str(field);
            output.push_str(")\n");
            indent(output, indent_level + 1);
            output.push_str("place: ");
            dump_rexpr(output, expr, module, indent_level + 1);
        }
        RExprKind::ArrayElement(array, index) => {
            output.push_str("ARRAY_ELEMENT\n");
            indent(output, indent_level + 1);
            output.push_str("array: ");
            dump_rexpr(output, array, module, indent_level + 1);
            indent(output, indent_level + 1);
            output.push_str("index: ");
            dump_expr(output, index, module, indent_level + 1);
        }
        RExprKind::Store(place, value) => {
            output.push_str("STORE\n");
            indent(output, indent_level + 1);
            output.push_str("place: ");
            dump_lexpr(output, place, module, indent_level + 1);
            indent(output, indent_level + 1);
            output.push_str("value: ");
            dump_expr(output, value, module, indent_level + 1);
        }
        RExprKind::GetPointer(expr) => {
            output.push_str("GET_POINTER ");
            dump_lexpr(output, expr, module, indent_level);
        }
        RExprKind::Block(e) => dump_block_expr(output, e, module, indent_level),
        RExprKind::Return(expr) => {
            output.push_str("RETURN ");
            dump_expr(output, expr, module, indent_level + 1);
        }
        RExprKind::Break(loop_id, expr) => {
            output.push_str("BREAK ");
            dump_expr(output, expr, module, indent_level + 1);
        }
        RExprKind::BinOp(binop, lhs, rhs) => {
            output.push_str("BIN_OP\n");
            indent(output, indent_level + 1);
            writeln!(output, "op: {binop:?}").unwrap();
            indent(output, indent_level + 1);
            output.push_str("lhs: ");
            dump_expr(output, lhs, module, indent_level + 1);
            indent(output, indent_level + 1);
            output.push_str("rhs: ");
            dump_expr(output, rhs, module, indent_level + 1);
        }
        RExprKind::If {
            cond,
            if_true,
            if_false,
        } => {
            output.push_str("IF\n");
            indent(output, indent_level + 1);
            output.push_str("cond: ");
            dump_expr(output, cond, module, indent_level + 1);
            indent(output, indent_level + 1);
            output.push_str("if_true: ");
            dump_expr(output, if_true, module, indent_level + 1);
            if let Some(if_false) = if_false {
                indent(output, indent_level + 1);
                output.push_str("if_false: ");
                dump_expr(output, if_false, module, indent_level + 1);
            }
        }
        RExprKind::Loop(loop_id, expr) => {
            output.push_str("LOOP\n");
            indent(output, indent_level + 1);
            output.push_str("body: ");
            dump_expr(output, expr, module, indent_level + 1);
        }
        RExprKind::ArrayInitializer(exprs) => output.push_str("ArrayInitializer\n"),
        RExprKind::StructInitializer(items) => output.push_str("StructInitializer\n"),
        RExprKind::FunctionCall(function_id, exprs) => {
            output.push_str("FUNCTION_CALL\n");
            indent(output, indent_level + 1);
            output.push_str("name: ");
            output.push_str(&module.functions.get(function_id).unwrap().name.value);
            output.push('\n');
            for arg in exprs {
                indent(output, indent_level + 1);
                output.push_str("arg: ");
                dump_expr(output, arg, module, indent_level + 1);
            }
        }
        RExprKind::Cast(expr) => output.push_str("Cast\n"),
        RExprKind::Not(expr) => output.push_str("Not\n"),
    }
}

fn dump_lexpr(output: &mut String, expr: &LExpr, module: &Module, indent_level: u32) {
    match &expr.kind {
        LExprKind::Dereference(expr) => {
            output.push_str("DEREFERENCE ");
            dump_expr(output, expr, module, indent_level);
        }
        LExprKind::Variable(variable_id) => {
            writeln!(output, "{variable_id:?}").unwrap();
        }
        LExprKind::Field(expr, field) => {
            output.push_str("FIELD_ACCESS (");
            output.push_str(field);
            output.push_str(")\n");
            indent(output, indent_level + 1);
            output.push_str("place: ");
            dump_lexpr(output, expr, module, indent_level + 1);
        }
        LExprKind::ArrayElement(array, index) => {
            output.push_str("ARRAY_ELEMENT\n");
            indent(output, indent_level + 1);
            output.push_str("array: ");
            dump_lexpr(output, array, module, indent_level + 1);
            indent(output, indent_level + 1);
            output.push_str("index: ");
            dump_expr(output, index, module, indent_level + 1);
        }
    }
}

fn indent(output: &mut String, indent: u32) {
    for _ in 0..indent {
        output.push(' ');
        output.push(' ');
        output.push(' ');
        output.push(' ');
    }
}
