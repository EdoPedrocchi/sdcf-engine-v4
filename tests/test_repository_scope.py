"""Keep the advisor-facing repository independent of empirical infrastructure."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_contains_no_private_or_internal_scope_markers() -> None:
    """Reject accidental personal paths, internal decisions, or empirical names."""
    forbidden = (
        "/" + "Users" + "/",
        "SDCF" + "_DATA_ROOT",
        "M" + "KC",
        "B" + "KNG",
        "D" + "-0",
    )
    checked = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and path.suffix in {".py", ".md", ".yaml", ".toml", ".yml"}
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in forbidden), path
