from ai_unifier_assesment.app import main


def test_main(capsys):
    """Test that main runs and prints output."""
    main()
    captured = capsys.readouterr()
    assert "App is running!" in captured.out
