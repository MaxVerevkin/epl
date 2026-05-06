mod ast;
mod common;
mod diagnostics;
mod entity_id;
mod ir;
mod ir_tree;
mod lex;
mod llvm;

fn main() {
    let mut args = std::env::args();
    let arg0 = args.next().expect("arg0 is always present");
    let Some(command) = args.next() else {
        print_usage(&arg0, 1)
    };
    if command == "help" {
        print_usage(&arg0, 0);
    }
    let Some(file) = args.next() else { print_usage(&arg0, 1) };
    if args.next().is_some() {
        print_usage(&arg0, 1)
    }
    let src = std::fs::read_to_string(&file).unwrap();

    match &*command {
        "ast" => {
            println!("{:#?}", ast(&file, &src));
        }
        "ir_tree" => {
            println!("{}", ir_tree(&file, &src).dump());
        }
        "ir" => {
            println!("{:#?}", ir(&file, &src));
        }
        "cfg" => {
            print!("{}", ir::graphviz::graph(&ir(&file, &src)));
        }
        "llvm-ir" => {
            println!("{}", llvm_module(&file, &src));
        }
        "llvm-obj" => {
            llvm_module(&file, &src).compile().unwrap_or_else(|err| {
                eprintln!("LLVM compilation error: {}", err.to_string_lossy());
                std::process::exit(1);
            });
        }
        other => {
            println!("Unknown command {other:?}");
            println!();
            print_usage(&arg0, 1);
        }
    }
}

fn ast(file: &str, src: &str) -> ast::Ast {
    ast::Parser::new(src).parse().unwrap_or_else(|err| {
        diagnostics::print_error(file, src, err);
        std::process::exit(1);
    })
}

fn ir_tree(file: &str, src: &str) -> ir_tree::Module {
    ir_tree::Module::from_ast(&ast(file, src)).unwrap_or_else(|err| {
        diagnostics::print_error(file, src, err);
        std::process::exit(1);
    })
}

fn ir(file: &str, src: &str) -> ir::Ir {
    ir::Ir::from_ir_tree(&ir_tree(file, src)).unwrap_or_else(|err| {
        diagnostics::print_error(file, src, err);
        std::process::exit(1);
    })
}

fn llvm_module(file: &str, src: &str) -> llvm::LlvmModule {
    llvm::LlvmModule::from_ir(&ir(file, src)).unwrap_or_else(|err| {
        eprintln!("LLVM verification error: {}", err.to_string_lossy());
        std::process::exit(1);
    })
}

fn print_usage(arg0: &str, status_code: i32) -> ! {
    println!("Usage: {arg0} <command> [command args]");
    println!();
    println!("Commands:");
    println!("  - help");
    println!("  - ast <file>");
    println!("  - ir_tree <file>");
    println!("  - ir <file>");
    println!("  - cfg <file>");
    println!("  - llvm-ir <file>");
    println!("  - llvm-obj <file>");
    std::process::exit(status_code);
}
