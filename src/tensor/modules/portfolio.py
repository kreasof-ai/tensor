"""Bounded, ordered shape regions for an AOT module export."""

from tensor.modules.errors import ModuleError
from tensor.runtime.domain import validate_domain, check_domain
from tensor.runtime.signature import scalar_value


def validate(portfolio):
    required = {"symbols", "shape_domain", "regions"}
    if (not isinstance(portfolio, dict) or not required <= portfolio.keys()
            or portfolio.keys() - (required | {"evidence"})):
        raise ModuleError("portfolio needs symbols, shape_domain and regions")
    symbols, domain, regions = (portfolio[k] for k in ("symbols", "shape_domain", "regions"))
    if (not isinstance(symbols, dict) or not symbols or len(symbols) > 16
            or any(not isinstance(k, str) or not k.isidentifier()
                   or v not in ("int32", "int64", "uint32", "uint64")
                   for k, v in symbols.items())):
        raise ModuleError("invalid portfolio dimension symbols")
    try:
        validate_domain(domain, symbols)
    except ValueError as error:
        raise ModuleError(str(error)) from error
    if not isinstance(regions, list) or not 1 <= len(regions) <= 64:
        raise ModuleError("portfolio needs 1..64 ordered regions")
    # Normalize to indexes of admissible multiples. Coverage is conservative:
    # it covers the entire bounding box, even when relations exclude some cells.
    axes = tuple(symbols)
    whole = tuple(((b["min"] + b.get("multiple_of", 1) - 1) // b.get("multiple_of", 1),
                   b["max"] // b.get("multiple_of", 1))
                  for b in (domain["bounds"][name] for name in axes))
    uncovered = [whole]
    for region in regions:
        bounds = region.get("bounds") if isinstance(region, dict) else None
        if not isinstance(bounds, dict) or bounds.keys() != symbols.keys():
            raise ModuleError("each region must bound every portfolio symbol")
        box = []
        for name in axes:
            bound = bounds[name]
            if (not isinstance(bound, dict) or set(bound) != {"min", "max"}
                    or any(type(v) is not int for v in bound.values())
                    or not domain["bounds"][name]["min"] <= bound["min"] <= bound["max"]
                    <= domain["bounds"][name]["max"]):
                raise ModuleError(f"invalid portfolio region bounds for {name}")
            step = domain["bounds"][name].get("multiple_of", 1)
            first, last = (bound["min"] + step - 1) // step, bound["max"] // step
            if first > last:
                raise ModuleError("portfolio region contains no admissible shape")
            box.append((first, last))
        remaining = []
        for cell in uncovered:
            remaining.extend(_subtract(cell, box))
            if len(remaining) > 4096:
                raise ModuleError("portfolio coverage exceeds the 4096-cell validation budget")
        uncovered = remaining
    if uncovered:
        raise ModuleError("portfolio regions leave a hole in the declared shape domain")


def _subtract(cell, box):
    intersection = tuple((max(a, c), min(b, d)) for (a, b), (c, d) in zip(cell, box))
    if any(a > b for a, b in intersection):
        return [cell]
    current, result = list(cell), []
    for axis, (low, high) in enumerate(intersection):
        first, last = current[axis]
        if first < low:
            piece = current.copy()
            piece[axis] = (first, low - 1)
            result.append(tuple(piece))
        if high < last:
            piece = current.copy()
            piece[axis] = (high + 1, last)
            result.append(tuple(piece))
        current[axis] = (low, high)
    return result


def select(portfolio, dimensions):
    if not isinstance(dimensions, dict) or dimensions.keys() != portfolio["symbols"].keys():
        raise ModuleError("portfolio resolution needs every concrete dimension")
    try:
        dimensions = {name: scalar_value(dimensions[name], dtype, name)
                      for name, dtype in portfolio["symbols"].items()}
        check_domain(portfolio["shape_domain"], dimensions)
    except ValueError as error:
        raise ModuleError(str(error)) from error
    for index, region in enumerate(portfolio["regions"]):
        if all(bound["min"] <= dimensions[name] <= bound["max"]
               for name, bound in region["bounds"].items()):
            return index, {k: v for k, v in region.items() if k != "bounds"}
    raise ModuleError("no portfolio region admits these dimensions")


def check_artifact(portfolio, region, manifest):
    """An image may be shared by regions, but must guard all assumptions."""
    contract = manifest.get("shape_domain")
    if manifest.get("symbols") != portfolio["symbols"] or contract is None:
        raise ModuleError("portfolio artifact needs the declared guarded dimension symbols")
    domain = portfolio["shape_domain"]
    if contract["relations"] != domain["relations"]:
        raise ModuleError("portfolio artifact relations disagree")
    for name, bound in region["bounds"].items():
        expected, actual = domain["bounds"][name], contract["bounds"][name]
        step = expected.get("multiple_of", 1)
        first, last = (bound["min"] + step - 1) // step * step, bound["max"] // step * step
        if (actual.get("multiple_of", 1) != step
                or actual["min"] > first or actual["max"] < last):
            raise ModuleError("portfolio artifact domain does not cover its dispatch region")
