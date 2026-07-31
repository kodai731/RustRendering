use crate::animation::editable::{EditableAnimationClip, SourceClip, SourceClipId};
use crate::ecs::component::ClipSchedule;
use crate::ecs::world::Entity;

#[derive(Clone, Debug)]
pub enum EditCommand {
    ClipModified {
        clip_id: SourceClipId,
        before: EditableAnimationClip,
        description: &'static str,
    },
    ScheduleModified {
        entity: Entity,
        before: ClipSchedule,
        description: &'static str,
    },
    ClipAdded {
        clip_id: SourceClipId,
        description: &'static str,
    },
    ClipRemoved {
        clip_id: SourceClipId,
        removed: SourceClip,
        description: &'static str,
    },
}

#[derive(Clone, Debug)]
pub enum EditCommandAfter {
    Clip(EditableAnimationClip),
    Schedule(ClipSchedule),
    ClipCreated(SourceClip),
    Empty,
}

#[derive(Clone, Debug)]
pub struct EditEntry {
    pub command: EditCommand,
    pub after: EditCommandAfter,
}

#[derive(Debug)]
pub struct EditHistory {
    undo_stack: Vec<EditEntry>,
    redo_stack: Vec<EditEntry>,
    max_history: usize,
}

impl EditHistory {
    pub fn new(max_history: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history,
        }
    }

    pub fn push_clip_edit(
        &mut self,
        clip_id: SourceClipId,
        before: EditableAnimationClip,
        after: EditableAnimationClip,
        description: &'static str,
    ) {
        let entry = EditEntry {
            command: EditCommand::ClipModified {
                clip_id,
                before,
                description,
            },
            after: EditCommandAfter::Clip(after),
        };
        self.push_entry(entry);
    }

    pub fn last_undo_mut(&mut self) -> Option<&mut EditEntry> {
        self.undo_stack.last_mut()
    }

    pub fn push_schedule_edit(
        &mut self,
        entity: Entity,
        before: ClipSchedule,
        after: ClipSchedule,
        description: &'static str,
    ) {
        let entry = EditEntry {
            command: EditCommand::ScheduleModified {
                entity,
                before,
                description,
            },
            after: EditCommandAfter::Schedule(after),
        };
        self.push_entry(entry);
    }

    pub fn push_clip_added(
        &mut self,
        clip_id: SourceClipId,
        source: SourceClip,
        description: &'static str,
    ) {
        let entry = EditEntry {
            command: EditCommand::ClipAdded {
                clip_id,
                description,
            },
            after: EditCommandAfter::ClipCreated(source),
        };
        self.push_entry(entry);
    }

    pub fn push_clip_removed(
        &mut self,
        clip_id: SourceClipId,
        removed: SourceClip,
        description: &'static str,
    ) {
        let entry = EditEntry {
            command: EditCommand::ClipRemoved {
                clip_id,
                removed,
                description,
            },
            after: EditCommandAfter::Empty,
        };
        self.push_entry(entry);
    }

    fn push_entry(&mut self, entry: EditEntry) {
        self.redo_stack.clear();
        self.undo_stack.push(entry);

        if self.undo_stack.len() > self.max_history {
            self.undo_stack.remove(0);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn pop_undo(&mut self) -> Option<EditEntry> {
        self.undo_stack.pop()
    }

    pub fn push_to_redo(&mut self, entry: EditEntry) {
        self.redo_stack.push(entry);
    }

    pub fn pop_redo(&mut self) -> Option<EditEntry> {
        self.redo_stack.pop()
    }

    pub fn push_to_undo(&mut self, entry: EditEntry) {
        self.undo_stack.push(entry);
    }

    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.last().map(|e| Self::entry_description(e))
    }

    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|e| Self::entry_description(e))
    }

    fn entry_description(entry: &EditEntry) -> &str {
        match &entry.command {
            EditCommand::ClipModified { description, .. } => description,
            EditCommand::ScheduleModified { description, .. } => description,
            EditCommand::ClipAdded { description, .. } => description,
            EditCommand::ClipRemoved { description, .. } => description,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_clip(name: &str) -> EditableAnimationClip {
        EditableAnimationClip::new(1, name.to_string())
    }

    #[test]
    fn test_push_and_undo_clip_edit() {
        let mut history = EditHistory::new(100);
        let before = make_dummy_clip("before");
        let after = make_dummy_clip("after");

        history.push_clip_edit(1, before, after, "test edit");

        assert!(history.can_undo());
        assert!(!history.can_redo());
        assert_eq!(history.undo_description(), Some("test edit"));

        let entry = history.pop_undo().unwrap();
        match &entry.command {
            EditCommand::ClipModified {
                clip_id,
                before,
                description,
            } => {
                assert_eq!(*clip_id, 1);
                assert_eq!(before.name, "before");
                assert_eq!(*description, "test edit");
            }
            _ => panic!("unexpected command type"),
        }
    }

    #[test]
    fn test_push_and_redo() {
        let mut history = EditHistory::new(100);
        let before = make_dummy_clip("before");
        let after = make_dummy_clip("after");

        history.push_clip_edit(1, before, after, "edit");

        let entry = history.pop_undo().unwrap();
        history.push_to_redo(entry);

        assert!(!history.can_undo());
        assert!(history.can_redo());
        assert_eq!(history.redo_description(), Some("edit"));
    }

    #[test]
    fn test_redo_cleared_on_new_edit() {
        let mut history = EditHistory::new(100);
        let before = make_dummy_clip("b1");
        let after = make_dummy_clip("a1");
        history.push_clip_edit(1, before, after, "first");

        let entry = history.pop_undo().unwrap();
        history.push_to_redo(entry);
        assert!(history.can_redo());

        let before2 = make_dummy_clip("b2");
        let after2 = make_dummy_clip("a2");
        history.push_clip_edit(1, before2, after2, "second");

        assert!(!history.can_redo());
    }

    #[test]
    fn test_max_history_limit() {
        let mut history = EditHistory::new(3);

        for i in 0..5 {
            let before = make_dummy_clip(&format!("b{}", i));
            let after = make_dummy_clip(&format!("a{}", i));
            history.push_clip_edit(1, before, after, "edit");
        }

        let mut count = 0;
        while history.pop_undo().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_last_undo_mut_returns_entry() {
        let mut history = EditHistory::new(100);
        let before = make_dummy_clip("before");
        let after = make_dummy_clip("after");
        history.push_clip_edit(1, before, after, "edit");

        assert!(history.last_undo_mut().is_some());
    }

    #[test]
    fn test_last_undo_mut_empty() {
        let mut history = EditHistory::new(100);
        assert!(history.last_undo_mut().is_none());
    }
}
