# ruff: noqa: INP001
# Standalone documentation sketch (not a package) — same convention as scripts/*.

"""
Programmatic access to the Windows channel: paramiko exec + SFTP.

Network modes mirror agent-linux/join_tailnet.sh:
- ``tun``       — system tailscaled with a TUN interface: paramiko connects to
                  the node's 100.x address directly.
- ``userspace`` — rootless tailscaled exposing a local SOCKS5 proxy: paramiko
                  dials through it (``SSHClient.connect(sock=...)``), which
                  requires PySocks (``uv run --with pysocks python <script>``).

Run one SSH transport per process invocation: the portable sshd on Windows
runs ``sshd -d`` (one connection per instance, restarted by the supervisor
loop), so a single client object carrying many channels is the right shape —
the same idea as ControlMaster in ssh_config.example, expressed in-process.
"""

import logging
import socket
from pathlib import Path

import paramiko

try:
    import socks  # PySocks, only needed for the userspace mode
except ImportError:  # pragma: no cover - environment-dependent optional extra
    socks = None

logger = logging.getLogger(__name__)

# ---------------------------- channel settings ------------------------------
CHANNEL_MODE = "tun"  # "tun" | "userspace"
WIN_HOST = "100.101.102.103"  # tailnet address of the Windows node
WIN_PORT = 2222  # $SshPort from windows/1-install.ps1
WIN_USER = "winuser"  # Windows account the channel runs under
AGENT_KEY = Path("~/.ssh/agent_ed25519").expanduser()  # agent private key
SOCKS_ADDR = ("127.0.0.1", 1055)  # userspace-mode SOCKS5 proxy (join_tailnet.sh)
# -----------------------------------------------------------------------------


def build_socket() -> socket.socket:
    """Return a socket for reaching ``WIN_HOST``, per ``CHANNEL_MODE``."""
    if CHANNEL_MODE == "tun":
        return socket.create_connection((WIN_HOST, WIN_PORT))
    if socks is None:
        msg = "userspace mode needs PySocks: uv run --with pysocks python paramiko_example.py"
        raise RuntimeError(msg)
    proxy = socks.socksocket(socket.AF_INET, socket.SOCK_STREAM)
    proxy.set_proxy(socks.SOCKS5, SOCKS_ADDR[0], SOCKS_ADDR[1])
    proxy.connect((WIN_HOST, WIN_PORT))
    return proxy


def connect() -> paramiko.SSHClient:
    """Open one SSH transport to the Windows node, strict host-key policy."""
    client = paramiko.SSHClient()
    # Host keys are pinned by the first interactive `ssh winbox` (see
    # ssh_config.example); this sketch refuses to trust unknown hosts.
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        WIN_HOST,
        port=WIN_PORT,
        username=WIN_USER,
        key_filename=str(AGENT_KEY),
        sock=build_socket(),
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def run_command(client: paramiko.SSHClient, command: str) -> int:
    """Execute ``command`` on the Windows node and stream its output."""
    stdin, stdout, stderr = client.exec_command(command)
    stdin.close()
    # NOTE: recv_exit_status() before draining stdout can deadlock on output
    # larger than the channel window — fine for the probe below; if this
    # sketch is reused for chatty commands, read stdout/stderr first.
    exit_code = stdout.channel.recv_exit_status()
    logger.info("exec %r -> exit %s", command, exit_code)
    out, err = stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")
    for line in out.splitlines():
        logger.info("stdout: %s", line)
    for line in err.splitlines():
        logger.warning("stderr: %s", line)
    return exit_code


def sftp_roundtrip(client: paramiko.SSHClient, local_file: Path) -> None:
    """Upload ``local_file`` to the node, then download it back as ``*.back``."""
    with client.open_sftp() as sftp:
        remote = "tsnet-roundtrip.bin"
        sftp.put(str(local_file), remote)
        sftp.get(remote, f"{local_file}.back")
        sftp.remove(remote)
    logger.info("sftp roundtrip OK: %s", local_file.name)


def main() -> int:
    """Connect, run a probe command, exercise SFTP, close cleanly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = connect()
    try:
        code = run_command(client, "git --version")
        sftp_roundtrip(client, Path(__file__))
    except (paramiko.SSHException, OSError):
        logger.exception("channel failure")
        return 1
    else:
        if code != 0:
            logger.error("probe command failed with exit code %s", code)
            return code
        logger.info("channel verified: exec + sftp both OK via %s mode", CHANNEL_MODE)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
