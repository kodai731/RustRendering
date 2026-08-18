use crate::animation::editable::{
    ClipGroup, ClipGroupId, ClipInstance, ClipInstanceId, SourceClipId,
};
use crate::ecs::component::ClipSchedule;

pub fn clip_schedule_add_instance(
    schedule: &mut ClipSchedule,
    source_id: SourceClipId,
    duration: f32,
) -> ClipInstanceId {
    let id = schedule.next_instance_id;
    schedule.next_instance_id += 1;
    let instance = ClipInstance::new(id, source_id, duration);
    schedule.instances.push(instance);
    id
}

pub fn clip_schedule_remove_instance(
    schedule: &mut ClipSchedule,
    instance_id: ClipInstanceId,
) -> bool {
    let before = schedule.instances.len();
    schedule.instances.retain(|i| i.instance_id != instance_id);

    for group in &mut schedule.groups {
        group.remove_instance(instance_id);
    }

    schedule.instances.len() < before
}

pub fn clip_schedule_active_instances(schedule: &ClipSchedule, time: f32) -> Vec<&ClipInstance> {
    schedule
        .instances
        .iter()
        .filter(|i| {
            if !i.is_active_at(time) {
                return false;
            }
            if let Some(group) = clip_schedule_find_group(schedule, i.instance_id) {
                return !group.muted;
            }
            true
        })
        .collect()
}

pub fn clip_schedule_create_group(schedule: &mut ClipSchedule, name: String) -> ClipGroupId {
    let id = schedule.next_group_id;
    schedule.next_group_id += 1;
    schedule.groups.push(ClipGroup::new(id, name));
    id
}

pub fn clip_schedule_remove_group(schedule: &mut ClipSchedule, group_id: ClipGroupId) {
    schedule.groups.retain(|g| g.id != group_id);
}

pub fn clip_schedule_add_to_group(
    schedule: &mut ClipSchedule,
    group_id: ClipGroupId,
    instance_id: ClipInstanceId,
) {
    for group in &mut schedule.groups {
        group.remove_instance(instance_id);
    }

    if let Some(group) = schedule.groups.iter_mut().find(|g| g.id == group_id) {
        group.add_instance(instance_id);
    }
}

pub fn clip_schedule_remove_from_group(
    schedule: &mut ClipSchedule,
    group_id: ClipGroupId,
    instance_id: ClipInstanceId,
) {
    if let Some(group) = schedule.groups.iter_mut().find(|g| g.id == group_id) {
        group.remove_instance(instance_id);
    }
}

pub fn clip_schedule_find_group(
    schedule: &ClipSchedule,
    instance_id: ClipInstanceId,
) -> Option<&ClipGroup> {
    schedule
        .groups
        .iter()
        .find(|g| g.contains_instance(instance_id))
}

/// Repoint the schedule's first instance at a newly selected clip. Reselecting
/// the clip the schedule already plays is a no-op, so user-trimmed `clip_out`
/// (e.g. a drag-extended flame clip) survives double-click / combo reselection.
pub fn clip_schedule_switch_source(
    schedule: &mut ClipSchedule,
    source_id: SourceClipId,
    duration: f32,
) {
    if let Some(first) = schedule.instances.first_mut() {
        if first.source_id == source_id {
            return;
        }
        first.source_id = source_id;
        first.clip_in = 0.0;
        first.clip_out = duration;
    }
}

pub fn clip_schedule_effective_weight(schedule: &ClipSchedule, instance_id: ClipInstanceId) -> f32 {
    let inst_weight = schedule
        .instances
        .iter()
        .find(|i| i.instance_id == instance_id)
        .map(|i| i.weight)
        .unwrap_or(0.0);

    match clip_schedule_find_group(schedule, instance_id) {
        Some(group) if group.muted => 0.0,
        Some(group) => inst_weight * group.weight,
        None => inst_weight,
    }
}

#[cfg(test)]
mod switch_source_tests {
    use super::*;

    fn schedule_with_instance(
        source_id: SourceClipId,
        clip_in: f32,
        clip_out: f32,
    ) -> ClipSchedule {
        let mut schedule = ClipSchedule::new();
        let mut inst = ClipInstance::new(1, source_id, 0.0);
        inst.clip_in = clip_in;
        inst.clip_out = clip_out;
        schedule.instances.push(inst);
        schedule
    }

    #[test]
    fn reselecting_same_source_preserves_trim() {
        let mut schedule = schedule_with_instance(3, 0.5, 3.0);
        clip_schedule_switch_source(&mut schedule, 3, 0.0);
        let inst = schedule.first_instance().unwrap();
        assert!((inst.clip_in - 0.5).abs() < 1e-6);
        assert!((inst.clip_out - 3.0).abs() < 1e-6);
    }

    #[test]
    fn switching_to_other_source_resets_range_to_new_duration() {
        let mut schedule = schedule_with_instance(3, 0.5, 3.0);
        clip_schedule_switch_source(&mut schedule, 7, 2.0);
        let inst = schedule.first_instance().unwrap();
        assert_eq!(inst.source_id, 7);
        assert!((inst.clip_in - 0.0).abs() < 1e-6);
        assert!((inst.clip_out - 2.0).abs() < 1e-6);
    }
}
