use std::path::{Path, PathBuf};
use std::{fmt, io};

struct TestCase {
    path: PathBuf,
    test_name: String,
}

impl fmt::Display for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.path.display(), self.test_name)
    }
}

#[derive(Default)]
struct Stats {
    ok: u32,
    fail: Vec<(TestCase, &'static str)>,
    new: Vec<TestCase>,
}

impl Stats {
    fn mark_ok(&mut self, test: TestCase) {
        println!("OK {test}");
        self.ok += 1;
    }

    fn mark_fail(&mut self, test: TestCase, reason: &'static str) {
        println!("FAIL {test}: {reason}");
        self.fail.push((test, reason));
    }

    fn mark_new(&mut self, test: TestCase) {
        println!("NEW {test}");
        self.new.push(test);
    }
}

fn main() {
    let mut args = std::env::args();
    let arg0 = args.next().unwrap();

    let Some(epl_exe) = args.next() else { usage(&arg0) };
    let Some(test_dir) = args.next() else { usage(&arg0) };

    if args.next().is_some() {
        usage(&arg0);
    }

    let mut stats = Stats::default();
    run_tests_in_dir(&mut stats, Path::new(&test_dir), &epl_exe);

    println!(
        "\nDONE ({} passed, {} failed, {} new)",
        stats.ok,
        stats.fail.len(),
        stats.new.len()
    );

    if !stats.new.is_empty() {
        println!("\nNew tests:");
        for test in &stats.new {
            println!("- {}/{}", test.path.display(), test.test_name);
        }
    }

    if !stats.fail.is_empty() {
        println!("\nFailed tests:");
        for (test, reason) in &stats.fail {
            println!("- {}/{}: {reason}", test.path.display(), test.test_name);
        }
    }

    if !stats.fail.is_empty() {
        std::process::exit(1);
    }
}

fn usage(arg0: &str) -> ! {
    println!("USAGE: {arg0} [epl exe] [test dir]");
    std::process::exit(1);
}

fn run_tests_in_dir(stats: &mut Stats, dir: &Path, epl_exe: &str) {
    for entry in std::fs::read_dir(dir).expect("could not read tests dir") {
        let entry = entry.expect("could not read entry in the tests dir");
        let path = entry.path();
        if path.is_dir() {
            run_tests_in_dir(stats, &path, epl_exe);
        } else if path.extension().is_some_and(|ext| ext == "epl") {
            run_tests_on_src(stats, &path, epl_exe);
        }
    }
}

fn run_tests_on_src(stats: &mut Stats, path: &Path, epl_exe: &str) {
    let test_case = TestCase {
        path: path.to_path_buf(),
        test_name: String::from("ir_tree"),
    };

    let result = std::process::Command::new(epl_exe)
        .arg("ir_tree")
        .arg(path)
        .output()
        .expect("compiler invocation failed");

    if !result.status.success() {
        stats.mark_fail(test_case, "compiler existed with non zero exit code");
        return;
    }

    let Ok(result_str) = String::from_utf8(result.stdout) else {
        stats.mark_fail(test_case, "compiler produced non-utf8 output");
        return;
    };

    let ir_tree_path = path.with_added_extension("ir_tree");
    match std::fs::read_to_string(&ir_tree_path) {
        Ok(expected_ir_tree) => {
            if expected_ir_tree == result_str {
                stats.mark_ok(test_case);
            } else {
                stats.mark_fail(test_case, "output does not match");
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            std::fs::write(ir_tree_path, result_str).expect("could not write expcted ir_tree output");
            stats.mark_new(test_case);
        }
        Err(e) => {
            panic!("could not read expected ir_tree output: {e:?}");
        }
    }
}
