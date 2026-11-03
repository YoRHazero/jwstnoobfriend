"""Tests for the shared JAX runtime configuration."""

from __future__ import annotations

import pytest

from noobfriend.core import jaxruntime


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("XLA_FLAGS", raising=False)
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
