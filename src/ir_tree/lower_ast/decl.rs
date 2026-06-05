use super::*;
use crate::ir_tree::types::*;

/// Get a type of a struct definition from its AST representation
pub fn lower_struct_decl(
    typesystem: &mut TypeSystem,
    type_namespace: &HashMap<String, Type>,
    ast: &ast::Struct,
    annotations: &[ast::Annotation],
) -> Result<Struct, Error> {
    if let Some(annotation) = annotations.iter().next() {
        return Err(
            Error::new(format!("unknown annotation: {:?}", annotation.ident.value)).with_span(annotation.span())
        );
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
