"""Steady-state allocation check after all model/operator states are warm."""

import gc
import hashlib
import time
from pathlib import Path
import torch
from tensor_torch.llt import Operators, AdamW
from .model import Config, Model
from .qualify import ROOT, OUT, setup, save, batch, train_step


def main():
    setup()
    ops = Operators(ROOT / "build/llt-qualification/artifacts")
    rows = []
    for architecture in ("llt", "naive"):
        model = Model(Config(architecture=architecture, checkpoint=True), ops).cuda()
        opt = AdamW(model.parameters(), ops)
        x, y = batch(torch.Generator().manual_seed(51), model.config)
        for _ in range(20):
            train_step(model, opt, x, y)
        torch.cuda.synchronize()
        gc.collect()
        samples = []
        for step in range(100):
            train_step(model, opt, x, y)
            if step % 10 == 0 or step == 99:
                torch.cuda.synchronize()
                gc.collect()
                samples.append(
                    {"step": step + 1, "allocated_bytes": torch.cuda.memory_allocated()}
                )
        growth = max(r["allocated_bytes"] for r in samples) - min(
            r["allocated_bytes"] for r in samples
        )
        assert growth <= 1024 * 1024, (architecture, growth)
        rows.append(
            {
                "architecture": architecture,
                "warmup_steps": 20,
                "steps": 100,
                "allowed_growth_bytes": 1024 * 1024,
                "measured_growth_bytes": growth,
                "samples": samples,
            }
        )
        del model, opt, x, y
        gc.collect()
        torch.cuda.empty_cache()
    source = Path(__file__)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot = OUT / "sources" / sha / source.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(source.read_bytes())
    save(
        "leak",
        {
            "status": "passed",
            "runs": rows,
            "source_sha256": sha,
            "coverage": ops.report,
        },
    )
    print(
        "Steady-state training allocation check passed:",
        [r["measured_growth_bytes"] for r in rows],
    )


if __name__ == "__main__":
    main()
