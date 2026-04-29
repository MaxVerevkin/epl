use std::fmt::Write;

use super::*;
use crate::ir_tree::visit::ExprVisitor;

pub fn dump(module: &Module) -> String {
    let mut writer = Writer {
        output: String::new(),
        module,
        function: None,
        indent_level: 1,
    };

    for function in module.functions.values() {
        writer.function = Some(function);
        writer.dump_function_desc(function);
        if let Some(body) = &function.body {
            writer.visit_expr(body);
        }
        writer.output.push('\n');
        writer.function = None;
    }

    writer.output
}

struct Writer<'a> {
    output: String,
    function: Option<&'a Function>,
    module: &'a Module,
    indent_level: u32,
}

impl ExprVisitor<'_> for Writer<'_> {
    fn visit_expr(&mut self, expr: &Expr) {
        self.indent();

        match &expr.kind {
            ExprKind::Const(value) => {
                self.output.push_str("CONST ");
                self.dump_constant(value);
            }
            ExprKind::ConstString(str) => write!(self.output, "CONST_STRING({str:?})").unwrap(),
            ExprKind::Load(_) => self.output.push_str("LOAD"),
            ExprKind::Field(_, field) => write!(self.output, "FIELD_ACCESS({field})").unwrap(),
            ExprKind::ArrayElement(_, _) => self.output.push_str("ARRAY_ELEMENT"),
            ExprKind::Store(_, _) => self.output.push_str("STORE"),
            ExprKind::GetPointer(_) => self.output.push_str("GET_POINTER"),
            ExprKind::Argument(arg) => {
                write!(self.output, "ARGUMENT({:?})", self.function.unwrap().args[*arg].0).unwrap();
            }
            ExprKind::Block(_) => self.output.push_str("BLOCK"),
            ExprKind::Return(_) => self.output.push_str("RETURN"),
            ExprKind::Break(loop_id, _) => write!(self.output, "BREAK({loop_id:?})").unwrap(),
            ExprKind::Arithmetic(op, _, _) => write!(self.output, "ARITHMETIC({op:?})").unwrap(),
            ExprKind::InPlaceArithmetic(op, _, _) => write!(self.output, "INPLACE_ARITHMETIC({op:?})").unwrap(),
            ExprKind::Cmp(op, _, _) => write!(self.output, "CMP({op:?})").unwrap(),
            ExprKind::If { .. } => self.output.push_str("IF"),
            ExprKind::Loop(loop_id, _) => write!(self.output, "LOOP({loop_id:?})").unwrap(),
            ExprKind::ArrayInitializer(_) => self.output.push_str("ARRAY_INITIALIZER"),
            ExprKind::StructInitializer(_) => self.output.push_str("STRUCT_INITIALIZER"),
            ExprKind::FunctionCall(function_id, _) => write!(
                self.output,
                "FUNCTION_CALL({:?})",
                self.module.functions.get(function_id).unwrap().name.value
            )
            .unwrap(),
            ExprKind::Cast(_) => self.output.push_str("CAST"),
            ExprKind::Not(_) => self.output.push_str("NOT"),
            ExprKind::Comptime(_) => self.output.push_str("COMPTIME"),
        }

        self.output.push_str(" TYPE=");
        self.dump_type(expr.ty);
        self.output.push('\n');

        self.indent_level += 1;
        if let ExprKind::Block(bexpr) = &expr.kind {
            for decl in &bexpr.variables {
                self.indent();
                write!(self.output, "DECLARE {:?} {:?} : ", decl.id, decl.debug_name).unwrap();
                self.dump_type(decl.ty);
                self.output.push('\n');
            }
        }
        expr.visit_children(self);
        self.indent_level -= 1;
    }

    fn visit_place(&mut self, place: &Place) {
        self.indent();

        match &place.kind {
            PlaceKind::Dereference(_) => self.output.push_str("DEREFERENCE"),
            PlaceKind::Variable(variable_id) => write!(self.output, "VARIABLE({variable_id:?})").unwrap(),
            PlaceKind::Field(_, field) => write!(self.output, "FIELD({field:?})").unwrap(),
            PlaceKind::ArrayElement(_, _) => self.output.push_str("ARRAY_ELEMENT"),
        }

        self.output.push_str(" [PLACE] TYPE=");
        self.dump_type(place.ty);
        self.output.push('\n');

        self.indent_level += 1;
        place.visit_children(self);
        self.indent_level -= 1;
    }
}

impl Writer<'_> {
    fn indent(&mut self) {
        if self.indent_level > 0 {
            self.output.push(' ');
            self.output.push(' ');
            self.output.push(' ');
            self.output.push(' ');
            for _ in 1..self.indent_level {
                self.output.push('|');
                self.output.push(' ');
                self.output.push(' ');
                self.output.push(' ');
            }
        }
    }

    fn dump_type(&mut self, ty: Type) {
        match ty {
            Type::Never => self.output.push('!'),
            Type::Unit => self.output.push_str("unit"),
            Type::Bool => self.output.push_str("bool"),
            Type::Int(int_type) => self.output.push_str(match int_type {
                IntType::I8 => "i8",
                IntType::U8 => "u8",
                IntType::I32 => "i32",
                IntType::U32 => "u32",
                IntType::I64 => "i64",
                IntType::U64 => "u64",
            }),
            Type::Struct(struct_id) => write!(self.output, "{struct_id:?}").unwrap(),
            Type::Ptr { pointee: None } => self.output.push_str("ptr"),
            Type::Ptr { pointee: Some(pointee) } => {
                self.output.push('*');
                self.dump_type(self.module.typesystem.get_type(pointee));
            }
            Type::Array { element, length } => {
                self.output.push('[');
                self.dump_type(self.module.typesystem.get_type(element));
                write!(self.output, "; {length}]").unwrap();
            }
        }
    }

    fn dump_function_desc(&mut self, function: &Function) {
        self.output.push_str("fn ");
        self.output.push_str(&function.name.value);
        self.output.push('(');
        for (arg_i, (arg_name, arg_ty)) in function.args.iter().enumerate() {
            self.output.push_str(arg_name);
            self.output.push_str(": ");
            self.dump_type(*arg_ty);
            if arg_i + 1 != function.args.len() || function.is_variadic {
                self.output.push_str(", ");
            }
        }
        if function.is_variadic {
            self.output.push_str("...");
        }
        self.output.push_str(") -> ");
        self.dump_type(function.return_ty);
        self.output.push('\n');
    }

    fn dump_constant(&mut self, value: &Constant) {
        match value {
            Constant::Undefined(_) => self.output.push_str("UNDEFINED"),
            Constant::Unit => self.output.push_str("UNIT"),
            Constant::Bool(bool) => write!(self.output, "{bool}").unwrap(),
            Constant::I8(num) => write!(self.output, "{num}").unwrap(),
            Constant::U8(num) => write!(self.output, "{num}").unwrap(),
            Constant::I32(num) => write!(self.output, "{num}").unwrap(),
            Constant::U32(num) => write!(self.output, "{num}").unwrap(),
            Constant::I64(num) => write!(self.output, "{num}").unwrap(),
            Constant::U64(num) => write!(self.output, "{num}").unwrap(),
            Constant::Array(_, elements) => {
                self.output.push('[');
                for (i, element) in elements.iter().enumerate() {
                    self.dump_constant(element);
                    if i + 1 != elements.len() {
                        self.output.push_str(", ");
                    }
                }
                self.output.push(']');
            }
        }
    }
}
