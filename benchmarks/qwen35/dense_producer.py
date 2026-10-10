"""AOT kernel bank for the dense Qwen local scaling exercise."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from tensor.compiler.entry import export_source
from tensor_llm.common.artifacts import identity
from tensor_llm.qwen35.dense.checkpoint import DenseCheckpoint
from tensor_llm.qwen35.dense.projections import schedule
from .build import build_artifact, needs_build


def _compile(job):
    entry, artifact, target = job
    build_artifact(entry, artifact, target=target)


def requirements(c, slots, chunk, pool, context, *, verify=False, draft=False):
    rows = slots * chunk
    kernels = {}

    def add(kind, **p):
        kernels[identity(kind, p)] = (kind, p)

    def linear(k, o, r=rows):
        p = schedule(r, k, o, c.width)
        add("split_linear", **p)
        add("split_merge", **p)

    common = dict(slots=slots, chunk=chunk, pool=pool)
    add("controls", **common)
    add("embedding", r=rows, c=c.width, vocab=c.vocab)
    for kind in ("rms", "add_rms"):
        add(kind, r=rows, c=c.width, eps=c.epsilon)
    proj = c.qkv_width + c.value_width + 2 * c.value_heads
    linear(c.width, proj)
    state = dict(commit=False) if verify else {}
    add("gdn_conv", **common, **state, channels=c.qkv_width, projection=proj)
    add(
        "gdn_prepare",
        **common,
        heads=c.value_heads,
        key_heads=c.key_heads,
        d=c.key_dim,
        channels=c.qkv_width,
        projection=proj,
    )
    add("gdn_scan", **common, **state, heads=c.value_heads, d=c.key_dim, tile=32)
    if verify:
        add("gdn_conv", **common, channels=c.qkv_width, projection=proj)
        add("gdn_scan", **common, heads=c.value_heads, d=c.key_dim, tile=32)
    add(
        "gdn_norm",
        **common,
        heads=c.value_heads,
        d=c.value_dim,
        channels=c.qkv_width,
        projection=proj,
        eps=c.epsilon,
    )
    linear(c.value_width, c.width)
    attproj = 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim
    linear(c.width, attproj)
    att = dict(
        common,
        heads=c.heads,
        kv_heads=c.kv_heads,
        d=c.head_dim,
        capacity=context,
        eps=c.epsilon,
    )
    add("attention_qkv", **att, theta=c.theta)
    add("attention", **att)
    linear(c.heads * c.head_dim, c.width)
    linear(c.width, 2 * c.intermediate)
    add("swiglu", **common, c=c.intermediate)
    linear(c.intermediate, c.width)
    add("last_rows", **common, c=c.width)
    head_rows = rows if verify else slots
    linear(c.width, c.vocab, r=head_rows)
    add("argmax", r=head_rows, vocab=c.vocab)
    add("head_linear", **schedule(head_rows, c.width, c.vocab, c.width))
    add("head_argmax", r=head_rows, tiles=(c.vocab + 63) // 64, vocab=c.vocab)
    if draft:
        add("mtp_join", r=rows, c=c.width, eps=c.epsilon)
        add("mtp_cast", r=rows, c=c.width)
        linear(2 * c.width, c.width)
        add("gather_hidden", **common, c=c.width)
        add("store_hidden", **common, c=c.width)
    return kernels


def source_hashes():
    from tensor_llm.qwen35.dense.artifacts import implementation

    return implementation()


def produce_control(out, *, slots, window, width=1024, target="sm_89"):
    """Compile the GPU control kernels without rebuilding model projections."""
    out = Path(out)
    rows = {}
    kinds = [
        ("spec_setup", {}),
        ("spec_accept", {}),
        ("spec_repair", {"width": width}),
        ("spec_result", {}),
    ]
    for depth in range(1, window - 1):
        kinds.extend(
            [("spec_draft", {"depth": depth}), ("spec_proposal", {"depth": depth})]
        )
    jobs = []
    for kind, extra in kinds:
        parameters = dict(slots=slots, window=window, **extra)
        key = identity(kind, parameters)
        text = export_source(
            "tensor_llm.qwen35.dense.spec_kernels",
            "make_kernel",
            kind,
            parameters,
            dependencies=("tensor.compiler.entry", "tensor.compiler.cuda_lowering"),
        )
        entry, artifact = out / (key + ".py"), out / (key + ".tbin")
        if needs_build(entry, artifact, text, target):
            artifact.unlink(missing_ok=True)
            entry.write_text(text)
            jobs.append((entry, artifact, target))
        rows[key] = dict(kind=kind, parameters=parameters, path=artifact.name)
    if jobs:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=4) as workers:
            list(workers.map(_compile, jobs))
    for row in rows.values():
        row["sha256"] = hashlib.sha256((out / row["path"]).read_bytes()).hexdigest()
    manifest = dict(
        schema="tensor.qwen35-dense-controls.v1",
        target=target,
        slots=slots,
        window=window,
        kernels=rows,
        implementation=source_hashes(),
    )
    (out / f"speculation-s{slots}-w{window}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def produce_recurrent(out, *, slots, window, pool=256, target="sm_89"):
    out = Path(out)
    common = dict(slots=slots, window=window, pool=pool)
    kinds = [
        ("defer_accept", {}),
        ("defer_clear", {}),
        ("conv_commit", dict(channels=6144, projection=8224)),
    ]
    kinds.extend(
        (kind, dict(heads=16, d=128, tile=32))
        for kind in ("gdn_deferred", "gdn_materialize")
    )
    rows = {}
    jobs = []
    for kind, extra in kinds:
        parameters = dict(common, **extra)
        key = identity(kind, parameters)
        text = export_source(
            "tensor_llm.qwen35.dense.recurrent_kernels",
            "make_kernel",
            kind,
            parameters,
            dependencies=("tensor.compiler.entry", "tensor.compiler.cuda_lowering"),
        )
        entry, artifact = out / (key + ".py"), out / (key + ".tbin")
        if needs_build(entry, artifact, text, target):
            artifact.unlink(missing_ok=True)
            entry.write_text(text)
            jobs.append((entry, artifact, target))
        rows[key] = dict(kind=kind, parameters=parameters, path=artifact.name)
    if jobs:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=4) as workers:
            list(workers.map(_compile, jobs))
    for row in rows.values():
        row["sha256"] = hashlib.sha256((out / row["path"]).read_bytes()).hexdigest()
    value = dict(
        schema="tensor.qwen35-recurrent-journal.v1",
        target=target,
        **common,
        kernels=rows,
        implementation=source_hashes(),
    )
    (out / f"recurrent-s{slots}-w{window}.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )


def produce(
    checkpoint,
    out,
    *,
    slots,
    chunk=1,
    pool=256,
    context=4096,
    target="sm_89",
    verify=False,
    draft=False,
):
    c = DenseCheckpoint(checkpoint).config
    if draft:
        from dataclasses import replace

        c = replace(c, layers=("full_attention",))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema="tensor.qwen35-dense.v1",
        config=asdict(c),
        slots=slots,
        chunk=chunk,
        pool=pool,
        context=context,
        target=target,
        verify=verify,
        draft=draft,
        kernels={},
    )
    pending = []
    for key, (kind, p) in requirements(
        c, slots, chunk, pool, context, verify=verify, draft=draft
    ).items():
        module = (
            "tensor_llm.qwen35.kernels.decode"
            if kind in ("embedding", "rms", "add_rms", "argmax")
            else (
                "tensor_llm.qwen35.dense.head"
                if kind in ("head_linear", "head_argmax")
                else (
                    "tensor_llm.qwen35.kernels.mtp"
                    if kind in ("mtp_join", "mtp_cast")
                    else (
                        "tensor_llm.qwen35.dense.projections"
                        if kind in ("split_linear", "split_merge")
                        else "tensor_llm.qwen35.dense.kernels"
                    )
                )
            )
        )
        text = export_source(
            module,
            "make_kernel",
            kind,
            p,
            dependencies=("tensor.compiler.entry", "tensor.compiler.cuda_lowering"),
        )
        entry = out / (key + ".py")
        artifact = out / (key + ".tbin")
        if needs_build(entry, artifact, text, target):
            artifact.unlink(missing_ok=True)
            entry.write_text(text)
            pending.append((entry, artifact, target))
    if pending:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=4) as workers:
            list(workers.map(_compile, pending))
    for key, (kind, p) in requirements(
        c, slots, chunk, pool, context, verify=verify, draft=draft
    ).items():
        artifact = out / (key + ".tbin")
        manifest["kernels"][key] = dict(
            kind=kind,
            parameters=p,
            path=artifact.name,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )
        print("built", kind, p, flush=True)
    manifest["implementation"] = source_hashes()
    name = f'{"mtp-" if draft else ""}inference-s{slots}-t{chunk}{"-verify" if verify else ""}.json'
    (out / name).write_text(json.dumps(manifest, indent=2) + "\n")
    return out / name


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--slots", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 32])
    p.add_argument("--pool", type=int, default=256)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--verify-chunks", type=int, nargs="*", default=[])
    p.add_argument("--mtp", action="store_true")
    a = p.parse_args()
    for slots in a.slots:
        for chunk in a.chunks:
            if chunk > 1 and slots > 32:
                continue
            produce(
                a.checkpoint,
                a.out,
                slots=slots,
                chunk=chunk,
                pool=a.pool,
                context=a.context,
            )
        for chunk in a.verify_chunks:
            produce(
                a.checkpoint,
                a.out,
                slots=slots,
                chunk=chunk,
                pool=a.pool,
                context=a.context,
                verify=True,
            )
        if a.mtp:
            for chunk in sorted(set([1, 32, *a.verify_chunks])):
                if chunk == 32 and slots > 32:
                    continue
                produce(
                    a.checkpoint,
                    a.out,
                    slots=slots,
                    chunk=chunk,
                    pool=a.pool,
                    context=a.context,
                    draft=True,
                )
            for window in a.verify_chunks:
                produce_control(
                    a.out,
                    slots=slots,
                    window=window,
                    target=a.target if hasattr(a, "target") else "sm_89",
                )
                produce_recurrent(a.out, slots=slots, window=window, pool=a.pool)
