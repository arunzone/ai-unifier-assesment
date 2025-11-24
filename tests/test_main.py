from unittest.mock import patch

from ai_unifier_assesment.app import main


def test_should_start_uvicorn_server():
    with patch("ai_unifier_assesment.app.uvicorn.run") as mock_run:
        main()

        mock_run.assert_called_once_with(
            "ai_unifier_assesment.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )
