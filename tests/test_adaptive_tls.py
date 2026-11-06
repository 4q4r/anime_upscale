"""Interception-adaptive TLS tests: retry-once semantics, banner dedup, strict mode, DoH purity."""

import inspect

import pytest
import requests

from anime_upscale import fetch_input as fi
from anime_upscale import gofile
from anime_upscale.gofile import GofileClient
from anime_upscale.reporting import OutputMode, Reporter
from anime_upscale.upload import pick_server

BANNER_VERBATIM = (
    "⚠ Перехват TLS на сети (self-signed цепочка) — соединение продолжено "
    "без проверки сертификата. Назначение подтверждено DoH-пиннингом; "
    "провайдер видит трафик. Отключить: [network] strict_tls = true"
)


def _cert_verify_error() -> requests.exceptions.SSLError:
    return requests.exceptions.SSLError(
        "HTTPSConnectionPool(host='api.gofile.io', port=443): "
        "Max retries exceeded with url: /servers "
        "(Caused by SSLError(SSLCertVerificationError(1, "
        "'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "self-signed certificate in certificate chain (_ssl.c:1010)')))"
    )


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Scripted sender: each call consumes the next outcome (exception or response)."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str = "", **kwargs: object) -> object:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _PinStub:
    def pin(self, hostname: str) -> str:
        return "203.0.113.7"


@pytest.fixture(autouse=True)
def _fresh_warning_state() -> None:
    gofile._intercept_warning_shown = False
    yield
    gofile._intercept_warning_shown = False


class TestAdaptiveRetry:
    def test_banner_constant_is_verbatim(self) -> None:
        assert gofile.INTERCEPT_WARNING == BANNER_VERBATIM

    def test_cert_verify_failure_retried_once_without_verification(self, capsys) -> None:
        session = _FakeSession([_cert_verify_error(), "response-ok"])
        response = gofile.adaptive_tls_request(session.get, url="https://api.gofile.io/servers")
        assert response == "response-ok"
        assert len(session.calls) == 2
        assert "verify" not in session.calls[0]
        assert session.calls[1]["verify"] is False
        assert BANNER_VERBATIM in capsys.readouterr().err

    def test_warning_printed_once_for_two_retries(self, capsys) -> None:
        session = _FakeSession([_cert_verify_error(), "r1", _cert_verify_error(), "r2"])
        gofile.adaptive_tls_request(session.get, url="https://a/one")
        gofile.adaptive_tls_request(session.get, url="https://a/two")
        assert len(session.calls) == 4
        assert session.calls[1]["verify"] is False
        assert session.calls[3]["verify"] is False
        assert capsys.readouterr().err.count(BANNER_VERBATIM) == 1

    def test_strict_tls_no_retry_error_names_flag_and_interception(self) -> None:
        session = _FakeSession([_cert_verify_error()])
        with pytest.raises(requests.exceptions.SSLError, match=r"strict_tls") as excinfo:
            gofile.adaptive_tls_request(
                session.get, url="https://api.gofile.io/servers", strict_tls=True
            )
        assert "перехват" in str(excinfo.value).lower()
        assert "[network] strict_tls = true" in str(excinfo.value)
        assert len(session.calls) == 1

    def test_non_ssl_error_no_retry_no_warning(self, capsys) -> None:
        session = _FakeSession([requests.exceptions.ConnectionError("connection refused")])
        with pytest.raises(requests.exceptions.ConnectionError):
            gofile.adaptive_tls_request(session.get, url="https://a/x")
        assert len(session.calls) == 1
        assert capsys.readouterr().err == ""

    def test_ssl_error_without_cert_verify_signature_no_retry(self, capsys) -> None:
        handshake = requests.exceptions.SSLError("TLS handshake failure: protocol version")
        session = _FakeSession([handshake])
        with pytest.raises(requests.exceptions.SSLError, match="handshake"):
            gofile.adaptive_tls_request(session.get, url="https://a/x")
        assert len(session.calls) == 1
        assert capsys.readouterr().err == ""


class TestGofileClientWiring:
    def test_fetch_contents_retries_cert_verify_failure(self) -> None:
        ok = _FakeResponse({"status": "ok", "data": {"type": "file", "children": None}})
        session = _FakeSession([_cert_verify_error(), ok])
        session.pin = _PinStub().pin  # type: ignore[method-assign]
        client = GofileClient(strict_tls=False)
        client.session = session  # type: ignore[assignment]
        client.token = "tok"
        payload = client.fetch_contents("abc")
        assert payload["status"] == "ok"
        assert len(session.calls) == 2
        assert session.calls[1]["verify"] is False

    def test_strict_client_loud_error_names_flag(self) -> None:
        session = _FakeSession([_cert_verify_error()])
        session.pin = _PinStub().pin  # type: ignore[method-assign]
        client = GofileClient(strict_tls=True)
        client.session = session  # type: ignore[assignment]
        client.token = "tok"
        with pytest.raises(gofile.GofileError, match=r"strict_tls"):
            client.fetch_contents("abc")


class TestRemoteNameWiring:
    def test_head_probe_retries_cert_verify_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        head_response = type("_Resp", (), {"status_code": 404, "headers": {}, "url": ""})()
        outcomes: list[object] = [_cert_verify_error(), head_response]
        head_calls: list[dict[str, object]] = []

        class _FakePinned:
            def __init__(self) -> None:
                self.calls = head_calls

            def pin(self, hostname: str) -> str:
                return "203.0.113.7"

            def head(self, url: str = "", **kwargs: object) -> object:
                self.calls.append({"url": url, **kwargs})
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        monkeypatch.setattr(fi, "PinnedSession", _FakePinned)
        assert fi._remote_name("https://host/get/file.mkv", strict_tls=False) == "file.mkv"
        assert len(head_calls) == 2
        assert head_calls[1]["verify"] is False


class TestUploadWiring:
    def test_pick_server_retries_cert_verify_failure(self) -> None:
        ok = _FakeResponse({"status": "ok", "data": {"servers": [{"name": "store1"}]}})
        session = _FakeSession([_cert_verify_error(), ok])
        session.pin = _PinStub().pin  # type: ignore[method-assign]
        client = GofileClient()
        client.session = session  # type: ignore[assignment]
        client.token = "tok"
        assert pick_server(client) == "store1"
        assert session.calls[1]["verify"] is False


class TestBuildQueueThreading:
    def test_strict_tls_threaded_to_default_prober(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, bool] = {}

        def fake_remote_name(url: str, strict_tls: bool = False) -> str:
            seen["strict_tls"] = strict_tls
            return "x.mkv"

        monkeypatch.setattr(fi, "_remote_name", fake_remote_name)
        reporter = Reporter(OutputMode(color=False, tty=False))
        queue = fi.build_queue(
            ["https://host/x.mkv"],
            lambda: None,  # type: ignore[arg-type, return-value]
            reporter,
            strict_tls=True,
        )
        assert [item.name for item in queue] == ["x.mkv"]
        assert seen["strict_tls"] is True


class TestDoHStaysStrict:
    def test_resolve_doh_never_disables_verification(self) -> None:
        source = inspect.getsource(gofile.resolve_doh)
        assert "verify=False" not in source
        assert "verify = False" not in source

    def test_resolve_doh_never_routes_through_adaptive_helper(self) -> None:
        source = inspect.getsource(gofile.resolve_doh)
        assert "adaptive_tls_request" not in source
        # F7: the ladder still talks via plain verified gets — now on a dedicated
        # trust_env=False session (env proxies would bypass the pinning policy).
        assert "session.get(" in source
