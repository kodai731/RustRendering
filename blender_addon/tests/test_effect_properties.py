"""Unit tests for group_params_by_owner group priority / owner fallback logic."""
from __future__ import annotations

from blender_addon.common.effect_properties import group_params_by_owner


def test_group_used_when_present_and_non_empty():
    """When 'group' is present and non-empty, it is used as the key."""
    params = [
        {"name": "a", "group": "crystal", "owner": "frame"},
        {"name": "b", "group": "crystal", "owner": "frame"},
        {"name": "c", "group": "wave", "owner": "frame"},
    ]
    result = group_params_by_owner(params)
    assert result == [("crystal", ["a", "b"]), ("wave", ["c"])]


def test_owner_used_when_group_missing():
    """When 'group' is missing, 'owner' is used as fallback."""
    params = [
        {"name": "a", "owner": "frame"},
        {"name": "b", "owner": "style"},
    ]
    result = group_params_by_owner(params)
    assert result == [("frame", ["a"]), ("style", ["b"])]


def test_owner_used_when_group_empty():
    """When 'group' is empty string, 'owner' is used as fallback."""
    params = [
        {"name": "a", "group": "", "owner": "frame"},
        {"name": "b", "group": "", "owner": "style"},
    ]
    result = group_params_by_owner(params)
    assert result == [("frame", ["a"]), ("style", ["b"])]


def test_default_owner_is_frame():
    """When both 'group' and 'owner' are missing, 'frame' is the default."""
    params = [
        {"name": "a"},
        {"name": "b"},
    ]
    result = group_params_by_owner(params)
    assert result == [("frame", ["a", "b"])]


def test_mixed_group_and_owner():
    """Mixed: some params have group, others fall back to owner."""
    params = [
        {"name": "a", "group": "crystal", "owner": "frame"},
        {"name": "b", "owner": "frame"},
        {"name": "c"},
    ]
    result = group_params_by_owner(params)
    assert result == [("crystal", ["a"]), ("frame", ["b", "c"])]
