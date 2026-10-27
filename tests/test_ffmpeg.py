"""ffmpeg mux/subprocess arg tests (plan §1/§5, research/05 §5)."""

from anime_upscale.ffmpeg import build_mux_args


class TestMuxArgs:
    def test_two_input_shape(self) -> None:
        args = build_mux_args(
            source="/proj/input/src.mkv",
            output="/proj/output/dst.mkv",
            video_args=["-c:v", "libx265", "-crf", "16"],
        )
        assert args[:2] == ["ffmpeg", "-y"]
        assert "-f" in args and args[args.index("-f") + 1] == "yuv4mpegpipe"
        assert args[args.index("-i") + 1] == "-"
        src_index = args.index("/proj/input/src.mkv")
        assert args[src_index - 1] == "-i"
        assert "-map" in args and args[args.index("-map") + 1] == "0:v:0"
        maps = [args[i + 1] for i, a in enumerate(args) if a == "-map"]
        assert maps == ["0:v:0", "1:a?", "1:s?", "1:t?"]
        assert args[args.index("-map_metadata") + 1] == "1"
        assert args[args.index("-map_chapters") + 1] == "1"
        assert args[args.index("-c:a") + 1] == "copy"
        assert args[args.index("-c:s") + 1] == "copy"
        assert args[args.index("-c:t") + 1] == "copy"
        assert args[-1] == "/proj/output/dst.mkv"

    def test_video_args_land_between_inputs_and_output(self) -> None:
        video = ["-c:v", "libx265", "-x265-params", "keyint=600"]
        args = build_mux_args(source="s.mkv", output="d.mkv", video_args=video)
        vpos = args.index("-c:v")
        srcpos = args.index("s.mkv")
        outpos = args.index("d.mkv")
        assert srcpos < vpos < outpos

    def test_stats_flag_present_for_progress(self) -> None:
        args = build_mux_args(source="s.mkv", output="d.mkv", video_args=[])
        assert "-stats" in args
