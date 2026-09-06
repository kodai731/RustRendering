use anyhow::Result;

use crate::app::App;

pub type EffectFrameHook = unsafe fn(&mut App, usize) -> Result<()>;
pub type EffectHookFn = unsafe fn(&mut App) -> Result<()>;

#[derive(Clone, Copy, Debug)]
pub struct EffectHook {
    pub name: &'static str,
    pub prepare_frame: Option<EffectFrameHook>,
    pub on_viewport_resize: Option<EffectHookFn>,
    pub destroy: Option<EffectHookFn>,
}

#[derive(Debug, Default)]
pub struct EffectHooks {
    entries: Vec<EffectHook>,
}

impl EffectHooks {
    pub fn register(&mut self, hook: EffectHook) {
        match self
            .entries
            .iter_mut()
            .find(|entry| entry.name == hook.name)
        {
            Some(entry) => *entry = hook,
            None => self.entries.push(hook),
        }
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.entries.iter().map(|entry| entry.name).collect()
    }

    fn snapshot(&self) -> Vec<EffectHook> {
        self.entries.clone()
    }
}

impl App {
    pub unsafe fn run_effect_prepare_frame(&mut self, frame_slot: usize) -> Result<()> {
        for hook in self.data.effect_hooks.snapshot() {
            if let Some(prepare_frame) = hook.prepare_frame {
                prepare_frame(self, frame_slot)?;
            }
        }
        Ok(())
    }

    pub unsafe fn run_effect_viewport_resize(&mut self) -> Result<()> {
        for hook in self.data.effect_hooks.snapshot() {
            if let Some(on_viewport_resize) = hook.on_viewport_resize {
                on_viewport_resize(self)?;
            }
        }
        Ok(())
    }

    pub unsafe fn run_effect_destroy(&mut self) -> Result<()> {
        for hook in self.data.effect_hooks.snapshot().into_iter().rev() {
            if let Some(destroy) = hook.destroy {
                destroy(self)?;
            }
        }
        Ok(())
    }
}
