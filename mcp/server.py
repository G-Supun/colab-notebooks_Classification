from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.classifier import SMSClassifier


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "model"
    / "best_model"
)


# ============================================================
# Load model once when the server starts
# ============================================================

classifier = SMSClassifier(
    model_dir=str(MODEL_DIR)
)


# ============================================================
# MCP Server
# ============================================================

mcp = FastMCP(
    "SMS Promotional Transactional Classifier"
)


# ============================================================
# Tool: classify_sms
# ============================================================

@mcp.tool()
def classify_sms(message: str) -> dict:
    """
    Classify an SMS message as promotional or transactional.

    Args:
        message: SMS message text.

    Returns:
        Classification category and confidence information.
    """

    return classifier.classify(message)


# ============================================================
# Tool: classify_batch
# ============================================================

@mcp.tool()
def classify_batch(messages: list[str]) -> list[dict]:
    """
    Classify multiple SMS messages.

    Args:
        messages: List of SMS message strings.

    Returns:
        Classification result for each message.
    """

    return classifier.classify_batch(messages)


# ============================================================
# Tool: get_model_info
# ============================================================

@mcp.tool()
def get_model_info() -> dict:
    """
    Return information about the deployed student model.
    """

    return classifier.model_info()


# ============================================================
# Start server
# ============================================================

if __name__ == "__main__":
    mcp.run()
