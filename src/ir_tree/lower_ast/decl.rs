use super::*;
use crate::ir_tree::types::*;

pub fn lower_function<'ctx>(
    ctx: Context<'ctx>,
    ast: &ast::Function,
    annotations: &[ast::Annotation],
    types_scope: &TypesScope<'ctx>,
) -> Result<Function<'ctx>, Error> {
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
        args.push((arg.name.value.clone(), type_from_ast(ctx, types_scope, &arg.ty)?));
    }

    Ok(Function {
        id: FunctionId::new(),
        name: ast.name.clone(),
        args,
        return_ty: ast
            .return_ty
            .as_ref()
            .map(|ty| type_from_ast(ctx, types_scope, ty))
            .transpose()?
            .unwrap_or_else(|| ctx.types().unit),
        is_variadic: ast.is_variadic,
        is_pure,
        body: None,
    })
}
