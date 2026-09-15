# Adaptive active training and sleep memory

AI-Lang v3.0.0 treats training as resumable bounded work. A program should not
need to allocate its complete dataset, model, and optimizer budget before it
can make progress. The same `.al` source can run in a short battery window or
as a long-lived desktop/server job.

## Device-aware work

`device_profile()` returns a map with the host platform, CPU count, available
memory, conservative `memory_budget`, `tier`, tensor `backend`, and a
`recommended_batch`. `batch_budget(item_bytes, requested)` estimates a batch
under 60 percent of best-effort available memory. This is deliberately
conservative and dependency-free; applications may choose a smaller budget
for thermal, battery, or latency reasons.

The profile is a planning hint, not a promise that a step function cannot
allocate additional memory. Keep the callback's temporary tensors bounded and
use an explicit `batch_size` when an item estimate is known.

## Active scheduler

```text
fn step(batch: List, state: Any) -> Map:
    # Update the model/optimizer in state and return the next state.
    give {"state": next_state, "loss": loss_value}.
done.

let result := active_train(data, step, "train.sleep", {
    "batch_size": 32,
    "epochs": 10,
    "max_items": 256,
    "max_seconds": 2.0,
    "checkpoint_every": 1,
    "state": initial_state
}).
```

The callback may also accept only `batch`; in that form it owns its state.
Returning a map with `state` replaces the persisted user state. Other returned
fields become the current metrics map. A non-map return is treated as the new
state when no prior state exists and is recorded as `result`.

The result map includes:

- `cursor`: the next data index in the current epoch;
- `epoch` and `epochs`;
- `items` and `steps` processed during this invocation;
- `complete` and `sleeping`;
- `batch_size`, `metrics`, `checkpoint`, and `device`.

`max_items` is an invocation budget. If it cuts through a batch, the final
batch is truncated exactly to the remaining item budget; completed items are
never silently overrun. `max_seconds` is checked between callbacks, so one
callback is allowed to finish atomically. At an epoch boundary the cursor is
reset to zero and the epoch is advanced. Empty data completes without calling
the callback.

On the next invocation, pass the same checkpoint path and data ordering. The
scheduler verifies the file, restores cursor/epoch/metrics/state, and begins
at the next chunk. It does not replay a completed chunk. A changed dataset
must be treated as a new run or rejected by application metadata; the generic
scheduler cannot infer dataset identity.

## Durable checkpoints and security

`sleep_save(path, state, cursor, metrics, epoch, metadata)` and
`sleep_load(path)` use the `AILANG-SLEEP-1` format. Saves write a temporary file
in the target directory, flush and `fsync` it, then atomically replace the
target. The JSON is bounded, rejects duplicate keys and non-finite numbers,
checks its schema, and has SHA-256 content and envelope hashes.

For an authenticated checkpoint, supply a key and key id:

```text
sleep_save("train.sleep", state, 12, {"loss": 0.3}, 1, {}, "secret", "phone-1").
let saved := sleep_load("train.sleep", "secret", true).
```

`active_train` accepts `signing_key`, `key_id`, and `require_signature` in its
options map. The secret is never written to the file. File capabilities remain
mandatory: grant only the checkpoint path with `fs.read` and `fs.write`.
Integrity checks do not replace an OS sandbox or authenticate an untrusted
program.

## Model and optimizer state

`model_save`/`model_load` persist nested model maps, records, and tensor leaves.
They accept the optional signing key/key id and required-signature flag in the
same positions as `sleep_save`/`sleep_load`. A `model_load` call with a
template reconstructs its record types:

```text
let restored := model_load("model.almodel", net_template).
```

Without a template, records become field-compatible maps; tensor leaves are
recreated as live tensors. `optimizer_save`/`optimizer_load` persist optimizer
step counts, hyperparameters, moments, and saved parameter values; they accept
the same
optional signing controls. Supplying fresh parameters to `optimizer_load`
rebinds the restored optimizer and
validates every moment shape. This avoids pickle and makes an AdamW checkpoint
portable between the pure-Python and accelerated tensor engines.

`packages/neural` builds on this boundary:

- `nn.active(net, rows, loss_fn, checkpoint, options)` trains rows shaped as
  `[features, target]`, saves both model and AdamW state in the active
  checkpoint, and rebinds optimizer parameters after resume;
- `nn.save`/`nn.load` use atomic model checkpoints while still reading the old
  JSON weights/biases format;
- `nn.save_bundle`/`nn.load_bundle` store model and optimizer files separately
  for deployment pipelines.

`packages/learning` exposes smaller wrappers around profiling, budgeting,
active scheduling, and model/optimizer persistence for custom training loops.
`packages/data` and `packages/metrics` keep common preparation and evaluation
operations in AI-Lang source, with no hidden I/O or required third-party
runtime.
