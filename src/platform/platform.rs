use imgui::{Context, FontConfig, FontGlyphRanges, FontSource};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::path::Path;
use winit::dpi::LogicalSize;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use super::clipboard;

pub struct System {
    pub event_loop: EventLoop<()>,
    pub window: Window,
    pub imgui: Context,
    pub platform: WinitPlatform,
}

pub fn init(title: &str, take_focus: bool, window_size: (u32, u32)) -> System {
    let title = match Path::new(&title).file_name() {
        Some(file_name) => file_name.to_str().unwrap_or(title),
        None => title,
    };
    let event_loop = EventLoop::new().expect("Failed to create EventLoop");

    #[allow(unused_mut)]
    let mut builder = WindowBuilder::new()
        .with_title(title)
        .with_active(take_focus)
        .with_inner_size(LogicalSize::new(window_size.0, window_size.1));
    // winit 0.29 ignores `with_active` on X11; an override-redirect window is
    // unmanaged by the WM and therefore can never steal input focus. It is also
    // always-on-top, so park it outside the visible screen — batch/MCP runs must
    // never cover the desktop or intercept the user's clicks.
    #[cfg(target_os = "linux")]
    if !take_focus {
        use winit::dpi::PhysicalPosition;
        use winit::platform::x11::WindowBuilderExtX11;
        builder = builder
            .with_override_redirect(true)
            .with_position(PhysicalPosition::new(10000, 10000));
    }
    let window = builder.build(&event_loop).expect("Failed to create window");

    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    imgui.io_mut().config_flags |= imgui::ConfigFlags::DOCKING_ENABLE;
    imgui.io_mut().backend_flags |= imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET;

    if let Some(backend) = clipboard::init() {
        imgui.set_clipboard_backend(backend);
    } else {
        eprintln!("Failed to initialize clipboard");
    }

    let mut platform = WinitPlatform::init(&mut imgui);
    {
        let dpi_mode = if let Ok(factor) = std::env::var("IMGUI_EXAMPLE_FORCE_DPI_FACTOR") {
            // Allow forcing of HiDPI factor for debugging purposes
            match factor.parse::<f64>() {
                Ok(f) => HiDpiMode::Locked(f),
                Err(e) => {
                    log_warn!("Invalid scaling factor '{}': {}, using default", factor, e);
                    HiDpiMode::Default
                }
            }
        } else {
            HiDpiMode::Default
        };

        platform.attach_window(imgui.io_mut(), &window, dpi_mode);
    }

    // Fixed font size. Note imgui_winit_support uses "logical
    // pixels", which are physical pixels scaled by the devices
    // scaling factor. Meaning, 13.0 pixels should look the same size
    // on two different screens, and thus we do not need to scale this
    // value (as the scaling is handled by winit)
    let font_size = 13.0;

    imgui.fonts().add_font(&[
        FontSource::TtfData {
            data: include_bytes!("../../assets/fonts/Roboto-Regular.ttf"),
            size_pixels: font_size,
            config: Some(FontConfig {
                // Oversampling font helps improve text rendering at
                // expense of larger font atlas texture.
                oversample_h: 4,
                oversample_v: 4,
                ..FontConfig::default()
            }),
        },
        FontSource::TtfData {
            data: include_bytes!("../../assets/fonts/mplus-1p-regular.ttf"),
            size_pixels: font_size,
            config: Some(FontConfig {
                // Oversampling font helps improve text rendering at
                // expense of larger font atlas texture.
                oversample_h: 4,
                oversample_v: 4,
                // Range of glyphs to rasterize
                glyph_ranges: FontGlyphRanges::japanese(),
                ..FontConfig::default()
            }),
        },
    ]);

    // Build the font atlas to generate texture data
    // This is required before any ImGui rendering can occur
    let _font_texture = imgui.fonts().build_rgba32_texture();

    System {
        event_loop,
        window,
        imgui,
        platform,
    }
}
