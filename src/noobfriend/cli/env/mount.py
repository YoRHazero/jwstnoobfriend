"""The ``env mount`` / ``env unmount`` commands: sshfs a remote root locally.

``mount`` sshfs-mounts ``NOOB_SERVER:DATA_ROOT_PATH`` (by default under the user
cache, outside any repo) and rewrites the ``.env`` so the library reads local
paths; ``unmount`` reverses it from the sidecar. The library itself stays
mount-unaware -- once mounted, it simply sees a local server.
"""

from pathlib import Path
from typing import Annotated

import typer

from noobfriend.cli.env._io import read_env_file, upsert_var
from noobfriend.cli.env._mount import (
    MountState,
    default_mountpoint,
    ensure_gitignored,
    git_work_tree,
    is_mounted,
    is_remote_server,
    is_tracked,
    load_state,
    plan_rewrite,
    remove_state,
    run_sshfs,
    run_unmount,
    save_state,
    sidecar_path,
    sshfs_available,
)
from noobfriend.core.console import console

_ENV_FILE_OPTION = typer.Option(
    "-f",
    "--env-file",
    help="The .env file to mount for. Defaults to ./.env.",
    resolve_path=True,
)


def _fail(message: str) -> None:
    """Print an error and abort the command."""
    console.print(f"[bold red]{message}[/bold red]")
    raise typer.Exit(code=1)


def cli_mount(
    env_file: Annotated[Path, _ENV_FILE_OPTION] = Path(".env"),
    path: Annotated[
        Path | None,
        typer.Option(
            "-p",
            "--path",
            help="Mountpoint to use instead of the default user-cache location.",
            resolve_path=True,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Remount over an existing/stale mount."),
    ] = False,
) -> None:
    """Mount NOOB_SERVER:DATA_ROOT_PATH locally and point the .env at it.

    The remote root is sshfs-mounted (by default under the user cache, outside
    any git repository) and NOOB_SERVER / DATA_ROOT_PATH / STAGE_*_PATH are
    rewritten to local paths, so the library reads through the mount. Run
    [cyan]env unmount[/cyan] to undo it.
    """
    if not env_file.exists():
        _fail(f".env not found at {env_file}. Run env init first.")
    env_dir = env_file.parent
    values = read_env_file(env_file)

    host = values.get("NOOB_SERVER")
    if not is_remote_server(host):
        _fail("NOOB_SERVER is local; there is nothing to mount.")
    host = host.strip()  # type: ignore[union-attr]
    data_root = values.get("DATA_ROOT_PATH")
    if not data_root:
        _fail("DATA_ROOT_PATH is not set in the .env file.")
    if load_state(env_dir) is not None and not force:
        _fail("Already mounted (sidecar present). Run env unmount first, or --force.")
    if not sshfs_available():
        _fail(
            "sshfs not found. Install it first "
            "(macOS: macFUSE + sshfs; Linux: the sshfs package)."
        )

    mountpoint = path or default_mountpoint(host, data_root)
    if is_mounted(mountpoint):
        if not force:
            _fail(f"{mountpoint} is already a mount. Use --force to remount.")
        run_unmount(mountpoint, force=True)

    proc = run_sshfs(host, data_root, mountpoint)
    if proc.returncode != 0:
        _fail(f"sshfs failed: {proc.stderr.decode(errors='replace').strip()}")

    saved, new, uncovered = plan_rewrite(values, mountpoint)
    for name, value in new.items():
        upsert_var(env_file, name, value)
    save_state(
        env_dir,
        MountState(
            host=host, data_root=data_root, mountpoint=str(mountpoint), saved=saved
        ),
    )

    _ignore_artifacts(env_dir, env_file, mountpoint)

    console.print(
        f"[bold green]Mounted [cyan]{host}:{data_root}[/cyan] at "
        f"[yellow]{mountpoint}[/yellow][/bold green]"
    )
    console.print(
        "Rewrote NOOB_SERVER / DATA_ROOT_PATH"
        + (f" / {len(new) - 2} stage path(s)" if len(new) > 2 else "")
        + " to local."
    )
    for name in uncovered:
        console.print(
            f"[yellow]{name} is not under DATA_ROOT_PATH; left remote.[/yellow]"
        )


def cli_unmount(
    env_file: Annotated[Path, _ENV_FILE_OPTION] = Path(".env"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Lazily/forcibly unmount even if the mount is busy.",
        ),
    ] = False,
) -> None:
    """Unmount the sshfs mount and restore the original .env values."""
    env_dir = env_file.parent
    state = load_state(env_dir)
    if state is None:
        console.print("[yellow]Not mounted (no sidecar found).[/yellow]")
        return

    mountpoint = Path(state.mountpoint)
    if is_mounted(mountpoint):
        proc = run_unmount(mountpoint, force=force)
        if proc.returncode != 0:
            _fail(
                f"unmount failed: {proc.stderr.decode(errors='replace').strip()}. "
                "Close anything using the mount, or retry with --force."
            )

    for name, value in state.saved.items():
        upsert_var(env_file, name, value)
    remove_state(env_dir)
    if mountpoint.exists() and not is_mounted(mountpoint):
        try:
            mountpoint.rmdir()
        except OSError:
            pass

    console.print(
        f"[bold green]Unmounted [yellow]{mountpoint}[/yellow] and restored "
        f"[cyan]{env_file}[/cyan].[/bold green]"
    )


def _ignore_artifacts(env_dir: Path, env_file: Path, mountpoint: Path) -> None:
    """Keep mount artifacts out of git: ignore the sidecar (and an in-repo mount)."""
    if git_work_tree(env_dir) is not None and ensure_gitignored(
        env_dir, sidecar_path(env_dir).name
    ):
        console.print(f"[dim]Added {sidecar_path(env_dir).name} to .gitignore.[/dim]")
    mount_tree = git_work_tree(mountpoint.parent)
    if mount_tree is not None and ensure_gitignored(
        mountpoint.parent, f"{mountpoint.name}/"
    ):
        console.print(
            f"[dim]Added {mountpoint.name}/ to {mountpoint.parent}/.gitignore "
            "(mountpoint is inside a git repo).[/dim]"
        )
    if is_tracked(env_file):
        console.print(
            f"[yellow]{env_file} is tracked by git; its rewritten (machine-local) "
            "paths should not be committed.[/yellow]"
        )
