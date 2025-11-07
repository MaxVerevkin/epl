# Syntax reference

- source = item*
- ident = [a-zA-Z_][a-zA-Z_0-9]* _except keywords_

## Comments

Everything after `#` is treated as a comment.

## Keywords

- `fn`
- `return`
- `if`
- `else`
- `loop`
- `let`
- `true`
- `false`

## Items

- item = function

## Fuctions

- function = `fn` ident `(` fn_arg? (, fn_arg)* `)` `->` ident (block_expr | `;`)
- fn_arg = ident `:` ident

## Expressions

- expr = additive_expr ( (`==` | `!=` | `<=` | `>=` | `<` | `>`) additive_expr )?
- additive_expr = multiplicative_expr ( (`+` | `-`) multiplicative_expr )*
- multiplicative_expr = base_expr ( (`*` | `/`) base_expr )*
- expr_with_block = block_expr | if_expr | loop_expr
- expr_with_no_block = expr _except exrp_with_block_
- base_expr = literal | function_call_expr | ident | `return` expr | `(` expr `)` | expr_with_block
- statement = `;` | let_statement | expr_with_no_block `;` | expr_with_block
- block_expr = `{` statement* expr_with_no_block? `}`
- if_expr = `if` expr block_expr (`else` block_expr)?
- loop_expr = `loop` block_expr
- function_call_expr = ident `(` expr? (, expr)* `)`
- let_statement = `let` ident `:` ident `;`

## Literals

- literal = number_literal | string_literal | bool_literal
- number_literal = [0-9]+
- string_literal = `"` [^"] `"`
- bool_literal = `true` | `false`
