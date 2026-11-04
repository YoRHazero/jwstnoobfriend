"""Tests for the shared JAX runtime configuration."""

from __future__ import annotations

import subprocess
import sys

import pytest

from noobfriend.core import jaxruntime


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv(jaxruntime.CPU_CORES_VARIABLE, raising=False)
    monkeypatch.setattr(jaxruntime, "read_environment", dict)
    monkeypatch.setattr(jaxruntime, "_backends_initialized", lambda: False)
    monkeypatch.setattr(jaxruntime, "_configured_too_late", False)
    return monkeypatch


class TestRequestedCores:
    """Parsing of ``NOOB_MCMC_CPU_CORES``."""

    def test_unset_means_every_core(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setattr(jaxruntime.os, "cpu_count", lambda: 6)
        assert jaxruntime.requested_cpu_cores() == 6

    def test_dotenv_value_is_read_without_environ(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setattr(
            jaxruntime, "read_environment", lambda: {"NOOB_MCMC_CPU_CORES": "3"}
        )
        assert jaxruntime.requested_cpu_cores() == 3

    @pytest.mark.parametrize("raw", ["0", "-2", "four"])
    def test_rejects_non_positive_integers(
        self, clean_env: pytest.MonkeyPatch, raw: str
    ) -> None:
        clean_env.setenv(jaxruntime.CPU_CORES_VARIABLE, raw)
        with pytest.raises(ValueError, match="NOOB_MCMC_CPU_CORES"):
            jaxruntime.requested_cpu_cores()


class TestConfigureHostDevices:
    """Writing the host-device count into ``XLA_FLAGS``."""

    def test_appends_flag_to_existing_flags(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")
        clean_env.setenv(jaxruntime.CPU_CORES_VARIABLE, "4")
        jaxruntime.configure_host_devices()
        assert jaxruntime.os.environ["XLA_FLAGS"] == (
            "--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=4"
        )

    def test_user_flag_wins(self, clean_env: pytest.MonkeyPatch) -> None:
        flags = "--xla_force_host_platform_device_count=2"
        clean_env.setenv("XLA_FLAGS", flags)
        clean_env.setenv(jaxruntime.CPU_CORES_VARIABLE, "4")
        jaxruntime.configure_host_devices()
        assert jaxruntime.os.environ["XLA_FLAGS"] == flags

    def test_initialized_backends_leave_flags_alone(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setattr(jaxruntime, "_backends_initialized", lambda: True)
        jaxruntime.configure_host_devices()
        assert "XLA_FLAGS" not in jaxruntime.os.environ
        assert jaxruntime._configured_too_late


def _run(script: str) -> str:
    """Run ``script`` in a fresh interpreter without a preset JAX_PLATFORMS."""
    prelude = "import os\nos.environ.pop('JAX_PLATFORMS', None)\n"
    return subprocess.run(
        [sys.executable, "-c", prelude + script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class TestRestrictToCpu:
    """Keeping a CPU-only process off accelerator backends."""

    def test_user_platforms_win(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("JAX_PLATFORMS", "cuda")
        jaxruntime.restrict_to_cpu()
        assert jaxruntime.os.environ["JAX_PLATFORMS"] == "cuda"

    def test_sets_environment_before_jax_is_imported(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setattr(jaxruntime, "_jax_imported", lambda: False)
        jaxruntime.restrict_to_cpu()
        assert jaxruntime.os.environ["JAX_PLATFORMS"] == "cpu"

    def test_sets_config_when_jax_is_imported_but_not_started(self) -> None:
        pytest.importorskip("jax")
        output = _run(
            "import jax\n"
            "from noobfriend.core.jaxruntime import restrict_to_cpu\n"
            "restrict_to_cpu()\n"
            "print(jax.config.jax_platforms, 'JAX_PLATFORMS' in os.environ)\n"
        )
        assert output == "cpu False"

    def test_started_backends_are_left_alone(self) -> None:
        pytest.importorskip("jax")
        output = _run(
            "import jax\n"
            "jax.devices()\n"
            "from noobfriend.core.jaxruntime import restrict_to_cpu\n"
            "restrict_to_cpu()\n"
            "print(jax.config.jax_platforms)\n"
        )
        assert output == "None"
