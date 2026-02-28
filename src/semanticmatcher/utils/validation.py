from typing import List, Dict, Any

__all__ = [
    "validate_entity",
    "validate_entities",
    "validate_threshold",
    "validate_model_name",
]


def validate_entity(entity: Dict[str, Any]) -> bool:
    """Validate a single entity dictionary."""
    if "id" not in entity:
        raise ValueError("Entity must have 'id' field")
    if "name" not in entity:
        raise ValueError("Entity must have 'name' field")
    return True


def validate_entities(entities: List[Dict[str, Any]]) -> bool:
    """Validate a list of entities."""
    if not entities:
        raise ValueError("entities list cannot be empty")

    for entity in entities:
        validate_entity(entity)

    ids = [e["id"] for e in entities]
    if len(ids) != len(set(ids)):
        raise ValueError("Entity IDs must be unique")

    return True


def validate_threshold(threshold: float) -> float:
    """Validate similarity threshold value."""
    if not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number")
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    return float(threshold)


def validate_model_name(model_name: str) -> bool:
    """Validate model name is not empty."""
    if not model_name or not isinstance(model_name, str):
        raise ValueError("model_name cannot be empty")
    return True
