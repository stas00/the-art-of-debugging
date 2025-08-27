# Debugging PyTorch

Most of the PyTorch debug notes are in my other book [here](https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md).

## Speeding up debug of large models

When debugging PyTorch workflows, as explained in [using small payload](../methodology#2-small-payload) you'd normally try to use tiny random models (see [here how to get and create those](https://github.com/stas00/ml-engineering/blob/master/debug/make-tiny-models-tokenizers-datasets.md). But since some problems only appear at scale it's very likely you'd have to use the full-sized model, which may take many minutes to load and run until it gets to the point of interest, where problems appear.

Given the nature of ML model architectures, they typically use a sequence of identical layers that repeat one after another. Therefore, if a model has say 48 layers, you can shrink it to just 2 layers, which will dramatically speed up both the loading and moving in the code. Of course, the qualitative outcome will be bad, but we aren't concerned with quality if the workload hangs or breaks.

One way to accomplish the model shrinking at the layer dimension is to clone the modeling repo and change `config.json` to a new number of layers, e.g. in HF hub's models edit `config.json`:
```
- "num_hidden_layers": 48,
+ "num_hidden_layers": 2,
```
and now load the model from the local cloned path. When the model gets loaded you will get a massive warning of unused weights, but you can ignore that.

Alternatively you can change the modeling code and hardcode the number of layers in the model's `__init__`. For example, if we use `Qwen3MoePreTrainedModel`:

```
class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        # add this line to load only 2 layers
        config.num_hidden_layers = 2
```

This is an easier approach than tweaking `config.json` if you plan to try various models of the same architecture, as you'd only need to tweak the modeling code once and not change the model configs.

If you want to load the full model, but only run a few layers, then you can hack the loop over the layers in the model's `forward`. If the original code in `Qwen3MoeModel.forward` was:

```
for decoder_layer in self.layers[: self.config.num_hidden_layers]):
    hidden_states = decoder_layer(...)
```
you can change to:
```
KEEP_N_LAYERS = 2
for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    # XXX: shortcut for much faster completion
    if idx+1 > KEEP_N_LAYERS: continue
    hidden_states = decoder_layer(...)
```

Now, let's say you work with a very long sequence length and full attention makes things too slow (because attention is quadratic in compute wrt sequence length). To get the memory allocation right you need to run it at least once (since all layers will use the same amount of memory). Same as with skipping layers, you can skip just attention runs. Let's say we run only attention in the last layer:

In attention `__init__` we set a few flags, let's use `Qwen3MoeAttention`:
```
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.skip_all_but_last_attention_debug_mode = True
        self.rotating_layer_counter = 0
```

and then in `Qwen3MoeAttention.forward`, we replace:
```
attn_output, attn_weights = attention_interface((self, query_states, ...)
```
(note the `...` - most args were trimmed for this exemplification), with:
```
import einops
if not self.skip_all_but_last_attention_debug_mode:
    attn_output, attn_weights = attention_interface(self, query_states, ...)
else:
    self.rotating_layer_counter = (self.rotating_layer_counter + 1) % self.num_hidden_layers
    # we detect the last layer by module counting since we know how many layers there are
    if self.rotating_layer_counter % self.num_hidden_layers == 0:
        attn_output, attn_weights = attention_interface(self, query_states, ...)
    else:
        # this feeds bogus data of the right shape connected to a graph - good enough for debug
        attn_output = einops.rearrange(query_states, "bs hc sl ... -> bs sl hcl ...")
        attn_weights = None
```
and install `pip install einops` for it to work.
