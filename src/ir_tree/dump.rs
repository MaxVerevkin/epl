use std::fmt::Write;

use crate::ir_tree::visit::ExprVisitor;

use super::*;

pub fn dump(module: &Module) -> String {
    let mut output = String::new();

    for function in module.functions.values() {
        dump_function_desc(&mut output, function, module);
        if let Some(body) = &function.body {
            dump_expr(&mut output, body, module, 1);
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
    output.push('\n');
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

fn dump_expr(output: &mut String, expr: &Expr, module: &Module, indent_level: u32) {
    indent(output, indent_level);

    match &expr.kind {
        ExprKind::Undefined => output.push_str("UNDEFINED"),
        ExprKind::ConstUnit => output.push_str("UNIT"),
        ExprKind::ConstNumber(num) => write!(output, "CONST_NUMBER({num})").unwrap(),
        ExprKind::ConstString(str) => write!(output, "CONST_STRING({str:?})").unwrap(),
        ExprKind::ConstBool(bool) => write!(output, "CONST_BOOL({bool})").unwrap(),
        ExprKind::Load(_) => output.push_str("LOAD"),
        ExprKind::Field(_, field) => write!(output, "FIELD_ACCESS({field})").unwrap(),
        ExprKind::ArrayElement(_, _) => output.push_str("ARRAY_ELEMENT"),
        ExprKind::Store(_, _) => output.push_str("STORE"),
        ExprKind::GetPointer(_) => output.push_str("GET_POINTER"),
        ExprKind::Argument(arg) => write!(output, "ARGUMENT({arg:?})").unwrap(),
        ExprKind::Block(_) => output.push_str("BLOCK"),
        ExprKind::Return(_) => output.push_str("RETURN"),
        ExprKind::Break(loop_id, _) => write!(output, "BREAK({loop_id:?})").unwrap(),
        ExprKind::Arithmetic(op, _, _) => write!(output, "ARITHMETIC({op:?})").unwrap(),
        ExprKind::Cmp(op, _, _) => write!(output, "CMP({op:?})").unwrap(),
        ExprKind::If { .. } => output.push_str("IF"),
        ExprKind::Loop(loop_id, _) => write!(output, "LOOP({loop_id:?})").unwrap(),
        ExprKind::ArrayInitializer(_) => output.push_str("ARRAY_INITIALIZER"),
        ExprKind::StructInitializer(_) => output.push_str("STRUCT_INITIALIZER"),
        ExprKind::FunctionCall(function_id, _) => write!(
            output,
            "FUNCTION_CALL({:?})",
            module.functions.get(function_id).unwrap().name.value
        )
        .unwrap(),
        ExprKind::Cast(_) => output.push_str("CAST"),
        ExprKind::Not(_) => output.push_str("NOT"),
    }

    output.push_str(" TYPE=");
    dump_type(output, expr.ty, module);
    output.push('\n');

    if let ExprKind::Block(bexpr) = &expr.kind {
        for (var_id, var_ty) in &bexpr.variables {
            indent(output, indent_level + 1);
            write!(output, "DECLARE {var_id:?} : ").unwrap();
            dump_type(output, *var_ty, module);
            output.push('\n');
        }
    }

    expr.visit_children(&mut Visitor {
        output,
        module,
        indent_level: indent_level + 1,
    });

    struct Visitor<'a> {
        output: &'a mut String,
        module: &'a Module,
        indent_level: u32,
    }

    impl ExprVisitor for Visitor<'_> {
        fn visit_expr(&mut self, expr: &Expr) {
            dump_expr(self.output, expr, self.module, self.indent_level);
        }

        fn visit_place(&mut self, place: &Place) {
            indent(self.output, self.indent_level);

            match &place.kind {
                PlaceKind::Dereference(_) => self.output.push_str("DEREFERENCE"),
                PlaceKind::Variable(variable_id) => write!(self.output, "VARIABLE({variable_id:?})").unwrap(),
                PlaceKind::Field(_, field) => write!(self.output, "FIELD({field:?})").unwrap(),
                PlaceKind::ArrayElement(_, _) => self.output.push_str("ARRAY_ELEMENT"),
            }

            self.output.push_str(" [PLACE] TYPE=");
            dump_type(self.output, place.ty, self.module);
            self.output.push('\n');

            self.indent_level += 1;
            place.visit_children(self);
            self.indent_level -= 1;
        }
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
