"""Lab mode: runs a small model locally with PyTorch and reads its real activations at every token."""
import base64
import threading
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

_lock = threading.Lock()
_model = None


def available():
    try:
        import torch, transformers  # noqa: F401
        return True
    except ImportError:
        return False


class Lab:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        # "eager" attention is the only implementation that returns attention weights
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager")
        self.model.eval()
        cfg = self.model.config
        self.layers, self.width = cfg.num_hidden_layers, cfg.intermediate_size
        self.heads, self.vocab, self.hidden = cfg.num_attention_heads, cfg.vocab_size, cfg.hidden_size
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.kv_bytes_per_token = 2 * self.layers * cfg.num_key_value_heads * head_dim * 4   # keys + values, float32
        self.params = sum(p.numel() for p in self.model.parameters())

        # Capture, for the newest position: each MLP neuron (input of down_proj) and the residual stream after each layer
        self.neurons = [None] * self.layers
        self.resid = [None] * (self.layers + 1)
        self.model.model.embed_tokens.register_forward_hook(lambda m, a, out: self._set_resid(0, out))
        for i, layer in enumerate(self.model.model.layers):
            layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, i=i: self.neurons.__setitem__(i, a[0][0, -1]))
            layer.register_forward_hook(lambda m, a, out, i=i: self._set_resid(i + 1, out[0] if isinstance(out, tuple) else out))
        self.special = set(self.tok.all_special_ids)

    def _set_resid(self, i, out):
        self.resid[i] = out[0, -1]

    def _token_text(self, i):
        return self.tok.decode([i])

    def generate(self, messages, stopped):
        """Yield one event per generated token with the real internal state of the model."""
        torch = self.torch
        from transformers import DynamicCache
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        context = ids[0].tolist()
        yield {"prompt_tokens": len(context), "context": [[self._token_text(i), i in self.special] for i in context]}
        past, generated, text = DynamicCache(), [], ""
        start = time.time()
        step_input = ids
        for _ in range(MAX_NEW_TOKENS):
            if stopped():
                return
            t0 = time.time()
            out = self.model(step_input, past_key_values=past, use_cache=True, output_attentions=True)
            step_ms = (time.time() - t0) * 1000
            past = out.past_key_values
            logits = out.logits[0, -1]
            probs = torch.softmax(logits, -1)

            # Sample with temperature + top-p
            p = torch.softmax(logits / TEMPERATURE, -1)
            sp, si = p.sort(descending=True)
            keep = sp.cumsum(0) - sp < TOP_P
            choice = int(si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)])

            event = self._snapshot(out, probs, choice, context)
            entropy = float(-(probs * probs.clamp_min(1e-12).log2()).sum())
            event.update({"id": choice, "tok": self._token_text(choice), "step_ms": round(step_ms, 1),
                          "input_tokens": step_input.shape[1], "context_len": len(context) + 1,
                          "kv_mb": round(self.kv_bytes_per_token * (len(context) + 1) / 2**20, 2),
                          "entropy": round(entropy, 2), "kept": int(keep.sum())})
            if choice in self.special:
                break
            generated.append(choice)
            context.append(choice)
            new_text = self.tok.decode(generated)
            event["token"] = new_text[len(text):]
            text = new_text
            yield event
            step_input = torch.tensor([[choice]])
        yield {"done": True, "eval_count": len(generated), "eval_duration": int((time.time() - start) * 1e9)}

    def _snapshot(self, out, probs, choice, context):
        torch = self.torch
        acts = torch.stack(self.neurons)                       # [layers, width]
        scale = acts.abs().quantile(0.995, dim=1).clamp_min(1e-6)
        q = (acts / scale[:, None] * 127).round().clamp(-127, 127).to(torch.int8)

        # How much each layer changes the representation
        resid = torch.stack(self.resid)
        delta = (resid[1:] - resid[:-1]).norm(dim=1)

        # Where the model looks: attention of the newest position, averaged over layers and heads
        att = torch.stack([a[0, :, -1, :] for a in out.attentions]).mean((0, 1))
        att_list = [(w, pos) for pos, w in enumerate(att.tolist())
                    if context[pos] not in self.special and self._token_text(context[pos]).strip()]
        att_list.sort(reverse=True)
        total = sum(w for w, _ in att_list) or 1
        attn = [[self._token_text(context[pos]), w / total] for w, pos in att_list[:5]]
        links = [[pos, round(w / total, 4)] for w, pos in att_list[:16]]   # for the attention sphere

        top_p, top_i = probs.topk(5)
        flat = acts.abs().flatten().topk(6).indices.tolist()
        hot = [[i // self.width, i % self.width, round(float(acts.flatten()[i]), 2)] for i in flat]

        return {
            "conf": float(probs[choice]),
            "top": [[self._token_text(int(i)), float(p)] for p, i in zip(top_p, top_i)],
            "neurons": base64.b64encode(q.numpy().tobytes()).decode(),
            "delta": [round(float(d), 2) for d in delta],
            "attn": attn,
            "links": links,
            "hot": hot,
        }


def get():
    """Load the model once (thread-safe)."""
    global _model
    with _lock:
        if _model is None:
            _model = Lab()
        return _model


def info():
    m = _model
    out = {"available": available(), "loaded": m is not None, "model": MODEL_NAME,
           "layers": m.layers if m else 24, "width": m.width if m else 4864}
    if m:
        out.update({"heads": m.heads, "vocab": m.vocab, "hidden": m.hidden, "params": m.params,
                    "temperature": TEMPERATURE, "top_p": TOP_P})
    return out
