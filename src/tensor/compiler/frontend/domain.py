"""Carry runtime-guarded shape facts into the existing TileLang lowering path."""

from tensor.runtime.domain import validate_domain


def constrain(kernel, domain, symbols):
    """Insert TIR assumptions, retaining dynamic loops and the original ABI.

    Callers serialize the same domain beside the executable. These assumptions
    are safe only when the runtime checks that domain before every binding.
    """
    if domain is None:
        return kernel
    validate_domain(domain, symbols)
    import tvm
    import tilelang
    from tvm import tirx

    variables = {}

    def visit(node):
        if isinstance(node, tirx.Var) and str(node.name) in symbols:
            name = str(node.name)
            if name in variables and not variables[name].same_as(node):
                raise ValueError(f"distinct frontend variables share dimension name {name}")
            variables[name] = node

    for parameter in kernel.params:
        visit(parameter)
    for buffer in kernel.buffer_map.values():
        for dimension in buffer.shape:
            tirx.stmt_functor.post_order_visit(dimension, visit)

    def expression(value):
        if type(value) is int:
            return tirx.IntImm("int64", value)
        if "var" in value:
            # The serialized expression evaluator uses checked int64 arithmetic;
            # do not accidentally introduce int32 overflow in an assumption.
            return tirx.Cast("int64", variables[value["var"]])
        if "cast" in value:
            return tirx.Cast(value["cast"], expression(value["value"]))
        left, right = (expression(argument) for argument in value["args"])
        operations = {
            "add": lambda: left + right, "sub": lambda: left - right,
            "mul": lambda: left * right, "floordiv": lambda: left // right,
            "floormod": lambda: left % right, "min": lambda: tirx.min(left, right),
            "max": lambda: tirx.max(left, right),
        }
        return operations[value["op"]]()

    facts = []
    for name, bound in domain["bounds"].items():
        variable = variables[name]
        facts.extend((variable >= bound["min"], variable <= bound["max"]))
        if bound.get("multiple_of", 1) != 1:
            facts.append(variable % bound["multiple_of"] == 0)
    for relation in domain["relations"]:
        left, right = expression(relation["left"]), expression(relation["right"])
        facts.append(left == right if relation["op"] == "eq" else left <= right)
    body = tirx.SeqStmt([*(tirx.Evaluate(tirx.assume(fact)) for fact in facts), kernel.body])
    constrained = kernel.with_body(body)
    module = tvm.IRModule({str(kernel.attrs["global_symbol"]): constrained})
    # Use TileLang's own scoped-assumption conversion before TVM Simplify.
    # Evaluate(assume(...)) alone is not consumed by the default simplifier.
    module = tilelang.transform.InjectAssumes()(module)
    return next(iter(tirx.transform.Simplify()(module).functions.values()))
