# Adaptive active training: each invocation does a bounded work window.
# Run the file again to resume from active_training.alstate. The checkpoint
# contains the network, AdamW moments, cursor, epoch, and current metrics.
#
# The exact same program is suitable for a phone battery window or a server;
# change max_items/max_seconds rather than rewriting the training loop.

use packages/neural as nn.

seed(23).
let rows := [
    [[0.0, 0.0], [1.0, 0.0]],
    [[0.0, 1.0], [0.0, 1.0]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 1.0], [1.0, 0.0]]
].
let net := nn.mlp_init([2, 4, 2], 23, "xavier").

fn loss_fn(logits: Any, target: Any) -> Any:
    give ce_t(logits, target).
done.

# A small max_items makes the sleep boundary visible. Run this script again
# until complete becomes true; every completed batch is skipped on resume.
let progress := nn.active(net, rows, loss_fn, "active_training.alstate", {
    "batch_size": 2,
    "max_items": 2,
    "epochs": 25,
    "rate": 0.03,
    "wd": 0.0,
    "checkpoint_every": 1
}).

emit "processed {progress.items} items; cursor {progress.cursor}; epoch {progress.epoch}".
emit "sleeping: {progress.sleeping}, complete: {progress.complete}".
when has(progress.metrics, "loss"):
    emit "last loss: {round(progress.metrics.loss, 6)}".
done.
