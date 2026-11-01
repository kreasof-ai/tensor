"""Compiler-independent bounded integer domains for executable shape symbols."""

from tensor.runtime.signature import INTEGER_TYPES, scalar_value, validate_expression, evaluate


def validate_domain(domain, symbols):
    """Validate the serialized contract without loading a frontend or solver.

    Every dynamic symbol has finite positive bounds. Relations use the same
    checked integer expressions as signatures; they do not infer missing values.
    """
    if (not isinstance(domain, dict) or set(domain) != {"version", "bounds", "relations"}
            or type(domain["version"]) is not int or domain["version"] != 1):
        raise ValueError("unsupported shape domain version or fields")
    bounds = domain["bounds"]
    if (not isinstance(bounds, dict) or not isinstance(symbols, dict) or not symbols
            or bounds.keys() != symbols.keys()
            or any(not isinstance(dtype, str) or dtype not in INTEGER_TYPES for dtype in symbols.values())):
        raise ValueError("shape domain must bound every dimension symbol")
    for name, bound in bounds.items():
        if (not isinstance(bound, dict) or set(bound) - {"min", "max", "multiple_of"}
                or not {"min", "max"} <= bound.keys()):
            raise ValueError(f"invalid shape bounds for {name}")
        low, high = bound["min"], bound["max"]
        step = bound.get("multiple_of", 1)
        if any(type(n) is not int for n in (low, high, step)) or not 1 <= low <= high or step < 1:
            raise ValueError(f"invalid positive shape bounds for {name}")
        scalar_value(low, symbols[name], name)
        scalar_value(high, symbols[name], name)
        scalar_value(high, "int64", name)
        if (low + step - 1) // step * step > high:
            raise ValueError(f"shape domain for {name} has no divisible value")
    relations = domain["relations"]
    if not isinstance(relations, list) or len(relations) > 128:
        raise ValueError("invalid shape domain relations")
    for relation in relations:
        if (not isinstance(relation, dict) or set(relation) != {"left", "op", "right"}
                or relation["op"] not in ("eq", "le")):
            raise ValueError("shape relations require left, eq/le and right")
        variables = validate_expression(relation["left"], symbols)
        variables.update(validate_expression(relation["right"], symbols))
        if not variables:
            left, right = evaluate(relation["left"], {}), evaluate(relation["right"], {})
            if not (left == right if relation["op"] == "eq" else left <= right):
                raise ValueError("shape domain contains a false constant relation")
    return domain


def check_domain(domain, symbols):
    """Reject inadmissible concrete dimensions before allocating or launching."""
    # Normalize NumPy integer scalars too: their arithmetic can wrap before the
    # checked expression evaluator sees an intermediate result.
    symbols = {name: scalar_value(evaluate({"var": name}, symbols), "int64", name)
               for name in domain["bounds"]}
    for name, bound in domain["bounds"].items():
        value = evaluate({"var": name}, symbols)
        if not bound["min"] <= value <= bound["max"]:
            raise ValueError(f"dimension {name}={value} is outside shape domain [{bound['min']}, {bound['max']}]")
        multiple = bound.get("multiple_of", 1)
        if value % multiple:
            raise ValueError(f"dimension {name}={value} must be divisible by {multiple}")
    for relation in domain["relations"]:
        left = evaluate(relation["left"], symbols)
        right = evaluate(relation["right"], symbols)
        if not (left == right if relation["op"] == "eq" else left <= right):
            raise ValueError(f"shape domain relation failed: {relation}")
