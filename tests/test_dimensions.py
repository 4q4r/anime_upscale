"""Mod-2 dimension guard decision tests (x265 yuv420 requires even dimensions)."""

import pytest

from anime_upscale.stages.dimensions import EvenAlign, decide_even_align


class TestDecideEvenAlign:
    def test_even_dimensions_are_noop(self) -> None:
        plan = decide_even_align(1920, 1080)
        assert plan.action == "none"
        assert (plan.width, plan.height) == (1920, 1080)

    def test_just_over_even_crops_one_pixel(self) -> None:
        # 321x241: odd on both axes, 1px over even — the live-suspect class.
        plan = decide_even_align(321, 241)
        assert plan.action == "crop"
        assert (plan.width, plan.height) == (320, 240)
        assert (plan.crop_right, plan.crop_bottom) == (1, 1)

    def test_single_odd_axis_crops_that_axis_only(self) -> None:
        plan = decide_even_align(1280, 721)
        assert plan.action == "crop"
        assert (plan.width, plan.height) == (1280, 720)
        assert (plan.crop_right, plan.crop_bottom) == (0, 1)

    def test_larger_deficit_scales(self) -> None:
        # Synthetic: budget tightened below the actual deficit → scale branch.
        plan = decide_even_align(321, 241, max_crop_px=1)
        assert plan.action == "scale"
        assert (plan.width, plan.height) == (320, 240)

    def test_scale_branch_never_upscales(self) -> None:
        plan = decide_even_align(323, 243, max_crop_px=0)
        assert plan.action == "scale"
        assert (plan.width, plan.height) == (322, 242)
        assert plan.width <= 323 and plan.height <= 243


class TestEvenAlignDataclass:
    def test_none_plan_has_zero_crops(self) -> None:
        plan = EvenAlign(action="none", width=640, height=360, crop_right=0, crop_bottom=0)
        assert plan.action == "none"

    def test_actions_are_closed_set(self) -> None:
        for action in ("none", "crop", "scale"):
            EvenAlign(action=action, width=2, height=2, crop_right=0, crop_bottom=0)
        with pytest.raises(ValueError, match="Недопустимое действие"):
            EvenAlign(action="pad", width=2, height=2, crop_right=0, crop_bottom=0)  # ty: ignore[invalid-argument-type]
