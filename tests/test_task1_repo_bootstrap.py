import importlib

import property_intel
from property_intel.config import REPO_ROOT, Settings, get_settings


def test_top_level_directories_exist() -> None:
    for name in ("data", "src", "tests", "docs", "notebooks"):
        assert (REPO_ROOT / name).is_dir()


def test_pyproject_and_env_example_exist() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / ".env.example").is_file()


def test_package_imports_and_has_version() -> None:
    importlib.reload(property_intel)
    assert isinstance(property_intel.__version__, str)
    assert property_intel.__version__


def test_settings_load_with_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.log_level == "INFO"


def test_settings_resolved_paths_are_absolute() -> None:
    settings = Settings(_env_file=None)
    assert settings.resolved_data_raw_dir().is_absolute()
    assert settings.resolved_data_processed_dir().is_absolute()
    assert settings.resolved_log_dir().is_absolute()


def test_get_settings_is_cached_singleton() -> None:
    assert get_settings() is get_settings()


def test_settings_override_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)
    assert settings.log_level == "DEBUG"
