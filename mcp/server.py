from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.classifier import SMSClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "model"
)


classifier = SMSClassifier(
    str(MODEL_DIR)
)


mcp = FastMCP(
    "SMS Classifier"
)


@mcp.tool()
def classify_sms(message: str) -> dict:
    """Classify an SMS as promotional or transactional."""

    return classifier.classify(message)


@mcp.tool()
def classify_batch(messages: list[str]) -> list[dict]:
    """Classify multiple SMS messages."""

    return classifier.classify_batch(messages)


@mcp.tool()
def get_model_info() -> dict:
    """Return information about the SMS classification model."""

    return classifier.model_info()


if __name__ == "__main__":
    mcp.run()
