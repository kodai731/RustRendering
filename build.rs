#![allow(non_snake_case)]
extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=./src/rust/lib.rs");

    // Cヘッダーファイルのパスを指定
    let header = "./src/cpp/AEInclude/AEBinding.hpp";
    let header_dir = PathBuf::from(header);
    //let headers_dir_canonical = canonicalize(header).unwrap();
    let include_path = header_dir.to_str().unwrap();
    // vulkan
    let vulkanHeader = "C:/VulkanSDK/1.2.162.1/include";
    let vulkanHeaderDir = PathBuf::from(vulkanHeader);
    let vulkanIncludePath = vulkanHeaderDir.to_str().unwrap();
    // glfw3
    let glfw3Header = "C:/Users/kodai/Projects/RustRendering/src/cpp/AEInclude/imgui/imgui/examples/libs/glfw/include/GLFW";
    let glfw3HeaderDir = PathBuf::from(glfw3Header);
    let glfw3IncludePath = glfw3HeaderDir.to_str().unwrap();
    // other
    let otherHeader = "C:/Users/kodai/Projects/RustRendering/src/cpp";
    let otherHeaderDir = PathBuf::from(otherHeader);
    let otherIncludePath = otherHeaderDir.to_str().unwrap();
    // glm
    let glmHeader = "C:/Users/kodai/Projects/RustRendering/src/cpp/glm";
    let glmHeaderDir = PathBuf::from(glmHeader);
    let glmIncludePath = glmHeaderDir.to_str().unwrap();
    // imgui
    let imguiHeader = "C:/Users/kodai/Projects/RustRendering/src/cpp/AEInclude/imgui/imgui";
    let imguiHeaderDir = PathBuf::from(imguiHeader);
    let imguiIncludePath = imguiHeaderDir.to_str().unwrap();    

    // bindgenを使ってRustコードを生成
    let bindings = bindgen::Builder::default()
        .header(header)
        .clang_arg(format!("-I{vulkanIncludePath}"))
        .clang_arg(format!("-I{glfw3IncludePath}"))
        .clang_arg(format!("-I{otherIncludePath}"))
        .clang_arg(format!("-I{glmIncludePath}"))
        .clang_arg(format!("-I{imguiIncludePath}"))
        .clang_arg(format!("-I{include_path}"))
        .clang_arg("-x")
        .clang_arg("c++")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // バインディングを出力する場所を設定
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}