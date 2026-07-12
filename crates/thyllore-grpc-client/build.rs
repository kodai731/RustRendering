fn main() {
    #[cfg(feature = "text-to-motion")]
    {
        println!("cargo:rerun-if-changed=proto/animation_ml.proto");

        let file_descriptors = protox::compile(&["proto/animation_ml.proto"], &["proto"])
            .expect("Failed to parse proto files");

        tonic_build::configure()
            .build_server(true)
            .compile_fds(file_descriptors)
            .expect("Failed to compile proto");
    }
}
