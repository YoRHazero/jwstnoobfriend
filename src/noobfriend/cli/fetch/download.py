"""The ``fetch download`` command: stream a product manifest to disk or a host."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel

from noobfriend.cli.fetch._download_service import download_products
from noobfriend.cli.fetch._io import load_products
from noobfriend.cli.fetch._presenters import (
    create_download_progress,
    create_search_progress,
    make_failed_downloads_table,
)
from noobfriend.cli.fetch._remote_service import (
    DownloadTarget,
    RemoteTransferError,
    probe_remote_mast,
    probe_ssh,
    relay_download,
    remote_download,
    resolve_target,
)
from noobfriend.cli.fetch._options import DownloadMode
from noobfriend.core.console import console


def cli_download(
    products_file: Annotated[
        Path,
        typer.Argument(
            help="File containing the products list to download, the output of [cyan]search[/cyan] command.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output",
            help="Destination: a local path, or a remote 'host:path' (host = an ~/.ssh/config alias; optional 'user@'). Defaults to the [yellow]NOOB_SERVER[/yellow] + stage path from .env, else ./downloads.",
            rich_help_panel="Output",
            metavar="[HOST:]PATH",
        ),
    ] = None,
    current_folder: Annotated[
        bool,
        typer.Option(
            "-c",
            "--current-folder",
            help="Download into the current folder. Overrides [blue]--output[/blue] and any environment default.",
            rich_help_panel="Output",
        ),
    ] = False,
    skip_exist: Annotated[
        bool,
        typer.Option(
            "-s",
            "--skip-exist",
            is_flag=True,
            help="Skip files that already exist with a matching size. By default existing files are overwritten.",
            rich_help_panel="Output",
        ),
    ] = False,
    mode: Annotated[
        DownloadMode,
        typer.Option(
            "--mode",
            help="Remote transfer mode: auto (probe, then choose), remote (force remote-direct), relay (force relaying through localhost). Ignored for local downloads.",
            rich_help_panel="Output",
        ),
    ] = DownloadMode.auto,
) -> None:
    """Download the JWST data listed in a products manifest, locally or to a host."""
    products = load_products(products_file)
    if not products:
        console.print("[red]No products found in the file.[/red]")
        raise typer.Exit(code=1)

    target = resolve_target(output, current_folder)
    _announce_destination(target, explicit=current_folder or output is not None)
    exist_mode = "skip" if skip_exist else "overwrite"

    if not target.is_remote:
        _download_local(products, Path(target.path), exist_mode)
    else:
        _download_remote(products, target, mode, skip_exist=skip_exist)


def _announce_destination(target: DownloadTarget, *, explicit: bool) -> None:
    """Show the download destination and, for an env-derived target, confirm.

    The destination is always printed so the user knows where the files will
    land before the time-consuming transfer starts. When it was not given
    explicitly (no ``-o``/``-c``) it came from the ``.env`` defaults — the
    surprising case, possibly a remote server path — so in an interactive session
    the user is asked to confirm and a declined prompt aborts. A non-interactive
    run (no TTY) skips the prompt and proceeds.

    Parameters
    ----------
    target : DownloadTarget
        The resolved destination.
    explicit : bool
        Whether the destination was given explicitly via ``-o`` or ``-c``.

    Raises
    ------
    typer.Abort
        If the user declines the confirmation prompt.
    """
    location = "remote" if target.is_remote else "local"
    source = "" if explicit else " [dim](from .env defaults)[/dim]"
    console.print(
        f"[bold]Download destination:[/bold] {target.display} [dim]({location})[/dim]{source}"
    )
    if explicit or not sys.stdin.isatty():
        return
    if not typer.confirm("Proceed with this destination?", default=True):
        raise typer.Abort


def _download_local(products: list[dict], output_dir: Path, exist_mode: str) -> None:
    """Download ``products`` into a local directory."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]{output_dir} is created[/green]")

    with create_download_progress() as progress:
        statuses = asyncio.run(
            download_products(products, output_dir, progress, exist_mode)
        )
    console.print(
        f"[green]Newly downloaded {sum(statuses)} out of {len(products)} products.[/green]"
    )


def _download_remote(
    products: list[dict],
    target: DownloadTarget,
    mode: DownloadMode,
    *,
    skip_exist: bool,
) -> None:
    """Download ``products`` to a remote host (remote-direct, or relay fallback)."""
    dest = target.ssh_dest
    assert dest is not None

    console.print(f"[teal]Connecting to {dest} ...[/teal]")
    ssh_error = asyncio.run(probe_ssh(dest))
    if ssh_error is not None:
        console.print(
            Panel(ssh_error, title=f"Cannot reach {dest}", border_style="red")
        )
        console.print(
            f"Check that [cyan]ssh {dest}[/cyan] works non-interactively: the host is in "
            "[yellow]~/.ssh/config[/yellow] and your key is loaded ([cyan]ssh-add -l[/cyan])."
        )
        raise typer.Exit(code=1)

    direct = _choose_remote_direct(dest, mode)

    try:
        if direct:
            console.print(
                f"[teal]Remote-direct download on {dest}:{target.path}[/teal]"
            )
            with create_search_progress() as progress:
                succeeded, failed = asyncio.run(
                    remote_download(
                        dest, target.path, products, progress, skip_exist=skip_exist
                    )
                )
        else:
            console.print(
                f"[teal]Relaying through localhost to {dest}:{target.path}[/teal]"
            )
            with create_download_progress() as progress:
                succeeded, failed = asyncio.run(
                    relay_download(
                        dest, target.path, products, progress, skip_exist=skip_exist
                    )
                )
    except RemoteTransferError as error:
        console.print(
            Panel(str(error), title="Remote transfer aborted", border_style="red")
        )
        raise typer.Exit(code=1) from error

    console.print(
        f"[green]{succeeded} out of {len(products)} products present on {dest}:{target.path}.[/green]"
    )
    if failed:
        console.print(make_failed_downloads_table(failed))
        raise typer.Exit(code=1)


def _choose_remote_direct(dest: str, mode: DownloadMode) -> bool:
    """Decide whether to use remote-direct, probing MAST reachability if needed."""
    if mode is DownloadMode.relay:
        return False
    if mode is DownloadMode.remote:
        if not asyncio.run(probe_remote_mast(dest)):
            console.print(
                Panel(
                    f"{dest} cannot reach MAST (curl missing or no outbound HTTPS).",
                    title="Remote-direct unavailable",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)
        return True

    direct = asyncio.run(probe_remote_mast(dest))
    if not direct:
        console.print(
            "[yellow]Remote cannot reach MAST directly; falling back to relay mode.[/yellow]"
        )
    return direct
