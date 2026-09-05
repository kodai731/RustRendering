use std::time::Instant;

use winit::event::Event;

use super::events::dispatch_window_event;
use super::key_bindings::default_bindings;
use super::ui::StatusBarState;
#[cfg(feature = "auto-rig")]
use super::ui::{TextToAnimationDialogState, TextToMeshDialogState};
use crate::app::App;
use crate::platform::System;

impl System {
    pub fn main_loop(self, app: &mut App) {
        let System {
            event_loop,
            window,
            mut imgui,
            mut platform,
        } = self;
        let mut last_frame = Instant::now();
        let bindings = default_bindings();
        let mut status_bar_state = StatusBarState::default();
        #[cfg(feature = "auto-rig")]
        let mut text_to_mesh_dialog_state = TextToMeshDialogState::default();
        #[cfg(feature = "auto-rig")]
        let mut text_to_animation_dialog_state = TextToAnimationDialogState::default();

        event_loop
            .run(move |event, window_target| match event {
                Event::NewEvents(_) => {
                    let now = Instant::now();
                    imgui.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;
                }

                Event::AboutToWait => {
                    platform
                        .prepare_frame(imgui.io_mut(), &window)
                        .expect("Failed to prepare frame");
                    window.request_redraw();
                }

                Event::WindowEvent {
                    event: ref window_event,
                    ..
                } => {
                    platform.handle_event(imgui.io_mut(), &window, &event);
                    dispatch_window_event(
                        window_event,
                        window_target,
                        app,
                        &mut imgui,
                        &mut platform,
                        &window,
                        &bindings,
                        &mut status_bar_state,
                        #[cfg(feature = "auto-rig")]
                        &mut text_to_mesh_dialog_state,
                        #[cfg(feature = "auto-rig")]
                        &mut text_to_animation_dialog_state,
                    );
                }

                Event::LoopExiting => {
                    unsafe { app.destroy() };
                }

                _ => {}
            })
            .expect("EventLoop error");
    }
}
