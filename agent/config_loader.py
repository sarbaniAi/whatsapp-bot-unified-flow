"""Config Loader — reads agent_config.yaml and resolves env vars."""

import os
import re
import yaml
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load agent config from YAML file, resolving ${ENV_VAR} placeholders."""
    if config_path is None:
        # Look in multiple locations
        for path in [
            os.path.join(os.path.dirname(__file__), "..", "config", "agent_config.yaml"),
            "/Workspace/Users/sarbani.maiti@databricks.com/whatsapp-agent/config/agent_config.yaml",
            "config/agent_config.yaml",
        ]:
            if os.path.exists(path):
                config_path = path
                break

    if not config_path or not os.path.exists(config_path):
        logger.warning(f"Config not found at {config_path}, using defaults")
        return _default_config()

    with open(config_path) as f:
        raw = f.read()

    # Resolve ${ENV_VAR} placeholders
    def _resolve(match):
        var = match.group(1)
        return os.environ.get(var, match.group(0))

    resolved = re.sub(r'\$\{(\w+)\}', _resolve, raw)
    config = yaml.safe_load(resolved)
    logger.info(f"Config loaded from {config_path}")
    return config


def _default_config():
    return {
        "agent": {"name": "WhatsApp Agent", "version": "1.0"},
        "llm": {"endpoint": "databricks-gpt-oss-120b", "max_tokens": 500, "temperature": 0.4},
        "mcp": {"enabled": False},
        "journey": {"steps": []},
        "whatsapp": {"provider": "twilio"},
        "eval": {"mlflow": {"trace_every_turn": True}},
    }


def get_field_config(config: dict, step_name: str) -> list[dict]:
    """Get field configurations for a journey step."""
    for step in config.get("journey", {}).get("steps", []):
        if step["name"] == step_name:
            return step.get("fields", [])
    return []


def get_all_fields(config: dict) -> list[dict]:
    """Get all fields across all steps in order."""
    fields = []
    for step in config.get("journey", {}).get("steps", []):
        for field in step.get("fields", []):
            field["step"] = step["name"]
            fields.append(field)
    return fields


def get_nudge_template(config: dict, step: str, lang: str = "hi") -> str:
    """Get nudge template for a drop-off step."""
    templates = config.get("nudge_templates", {})
    step_map = {
        "OTP_VERIFIED": "dropped_otp",
        "BASIC_DETAILS": "dropped_basic",
        "ELIGIBILITY_DETAILS": "dropped_eligibility",
        "LOAN_OFFER": "loan_offer_pending",
    }
    template_key = step_map.get(step, "dropped_otp")
    return templates.get(template_key, {}).get(lang, templates.get(template_key, {}).get("hi", ""))
