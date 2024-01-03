use std::os::raw::c_int;

#[no_mangle]
pub extern "C" fn RAdd(n: *mut c_int, m: *mut c_int) -> c_int {
    unsafe{
        let x = *n;
        let y = *m;
        x + y
    }
}

#[cfg(target_os = "android")]
pub mod android {
    extern crate jni;
    use self::jni::objects::{JClass, JString};
    use self::jni::sys::{jint, jstring};
    use self::jni::JNIEnv;
    use super::*;

    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aqoole_vulkanNativeActivity(
        env: JNIEnv,
        _: JClass,
        input: JString,
    ) -> jstring {
        let input: String = env
            .get_string(input)
            .expect("invalid pattern string")
            .into();
        let output = env
            .new_string(format!("Android meets {}", input))
            .expect("Couldn't create java string!");
        output.into_inner()
    }

    #[no_mangle]
    pub unsafe extern "C" fn Java_com_aqoole_VulkanNativeActivity_RAdd(
        _env: JNIEnv,
        _: JClass,
        java_int: jint,
    ) -> jint {
        let r = RAdd(java_int as *mut i32, java_int as *mut i32);
        r as jint
    }
}