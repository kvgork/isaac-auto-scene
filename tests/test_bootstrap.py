"""Bootstrap smoke tests — package importable, CLI entry point exists."""


def test_package_importable() -> None:
    import isaac_auto_scene

    assert isaac_auto_scene.__version__ == "0.1.0"


def test_cli_entry_point() -> None:
    from isaac_auto_scene.cli import main

    assert callable(main)
