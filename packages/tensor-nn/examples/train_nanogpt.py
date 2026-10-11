"""Train the nanoGPT example with an explicit Tensor forward/backward/update loop."""

import argparse
import numpy as np
import tensor
from tensor_nn_examples.nanogpt import NanoGPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", required=True, help="A precompiled nanoGPT training bundle"
    )
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    rng = np.random.default_rng(123)
    with tensor.Device() as device:
        model = NanoGPT(args.bundle, device)
        cfg = model.config
        for step in range(args.steps):
            data = rng.integers(
                0, cfg.vocab, (cfg.batch, cfg.sequence + 1), dtype=np.int32
            )
            model.set_batch(data[:, :-1], data[:, 1:])

            logits, model_context = model(model.tokens)
            losses, loss_context = model.criterion(logits, model.targets)
            logit_gradient, _ = model.criterion.backward(loss_context, model.loss_grad)
            model.backward(model_context, logit_gradient)
            model.optimizer.step()

            print(
                f"step {step + 1}: loss={model.read_loss():.6f}, norm={model.read_norm():.6f}"
            )


if __name__ == "__main__":
    main()
