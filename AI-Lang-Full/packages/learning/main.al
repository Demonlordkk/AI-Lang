# Universal active training.
#
# The scheduler is intentionally small at the language layer. Your step
# function owns the model and loss; active() owns adaptive batching,
# checkpointing and sleep/resume. The exact same source works on a phone,
# laptop, server or embedded Python host.
#
# A step function can be:
#   fn step(batch: List, state: Any) -> Map:
#       ...
#       give {"state": next_state, "loss": loss}.
#   done.
#
# Call active() repeatedly from a foreground task, cron job, battery window,
# or service. Set max_items/max_seconds to return control deliberately.

fn profile() -> Map:
    give device_profile().
done.

fn batch_size(item_bytes: Int, requested: Int) -> Int:
    give batch_budget(item_bytes, requested).
done.

fn active(data: List, step_fn: Function, checkpoint: Text, options: Map) -> Map:
    give active_train(data, step_fn, checkpoint, options).
done.

fn save(path: Text, state: Any, cursor: Int, metrics: Map, epoch: Int, metadata: Map) -> Map:
    give sleep_save(path, state, cursor, metrics, epoch, metadata).
done.

fn restore(path: Text) -> Map:
    give sleep_load(path).
done.

fn save_model(path: Text, model: Any, metadata: Map) -> Map:
    give model_save(path, model, metadata).
done.

fn load_model(path: Text, template: Any) -> Any:
    give model_load(path, template).
done.

fn save_optimizer(path: Text, optimizer: Map, metadata: Map) -> Map:
    give optimizer_save(path, optimizer, metadata).
done.

fn load_optimizer(path: Text, params: Any) -> Map:
    give optimizer_load(path, params).
done.

# Run one bounded window and return a compact progress summary suitable for a
# UI. It never hides the full scheduler result.
fn run_window(data: List, step_fn: Function, checkpoint: Text, max_items: Int, max_seconds: Real) -> Map:
    let options := {"max_items": max_items, "max_seconds": max_seconds, "checkpoint_every": 1}.
    let result := active(data, step_fn, checkpoint, options).
    give {"complete": result.complete, "sleeping": result.sleeping,
        "cursor": result.cursor, "epoch": result.epoch,
        "items": result.items, "batch_size": result.batch_size,
        "metrics": result.metrics}.
done.
