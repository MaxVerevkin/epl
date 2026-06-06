use super::*;
use crate::ir_tree::types::*;

pub fn lower_function(
    typesystem: &mut TypeSystem,
    type_namespace: &HashMap<String, Type>,
    ast: &ast::Function,
    annotations: &[ast::Annotation],
) -> Result<Function, Error> {
    let mut is_pure = false;
    for annotation in annotations {
        match annotation.ident.value.as_str() {
            "pure" => is_pure = true,
            _ => return Err(Error::unknown_annotation(annotation)),
        }
    }

    let mut args: Vec<(String, Type)> = Vec::new();
    for arg in &ast.args {
        if args.iter().any(|x| x.0 == arg.name.value) {
            return Err(Error::new("argument with this name already exists").with_span(arg.name.span));
        }
        args.push((
            arg.name.value.clone(),
            typesystem.type_from_ast(type_namespace, &arg.ty)?,
        ));
    }

    Ok(Function {
        id: FunctionId::new(),
        name: ast.name.clone(),
        args,
        return_ty: ast
            .return_ty
            .as_ref()
            .map(|ty| typesystem.type_from_ast(type_namespace, ty))
            .transpose()?
            .unwrap_or(Type::Unit),
        is_variadic: ast.is_variadic,
        is_pure,
        body: None,
    })
}

pub fn lower_struct(
    typesystem: &mut TypeSystem,
    type_namespace: &HashMap<String, Type>,
    ast: &ast::Struct,
    annotations: &[ast::Annotation],
) -> Result<Struct, Error> {
    if let Some(annotation) = annotations.iter().next() {
        return Err(Error::unknown_annotation(annotation));
    }

    let mut fields: Vec<StructField> = Vec::new();
    for field in &ast.fields {
        if fields.iter().any(|x| x.name.value == field.name.value) {
            return Err(Error::new("field with this name already exists").with_span(field.name.span));
        }
        let ty = typesystem.type_from_ast(type_namespace, &field.ty)?;
        fields.push(StructField {
            name: field.name.clone(),
            ty,
            offset: None,
        });
    }

    Ok(Struct {
        name: ast.name.clone(),
        fields,
        layout: None,
    })
}
