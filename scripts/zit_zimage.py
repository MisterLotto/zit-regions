"""
Z-Image (NextDiT / Lumina2-family) adapter for zit-regions.

What this knows that the core doesn't:
  - how to detect that the loaded model is Z-Image
  - which attention modules to monkeypatch (JointAttention)
  - how to encode each region's prompt and concatenate them into one caption
    sequence, recording each region's column span
  - the caption/image split point at attention time

Confirmed by reading the Forge backend:
  - backend/nn/lumina.py     : class NextDiT, class JointAttention.forward(x, x_mask, freqs_cis, transformer_options)
                               sequence is cat((cap_feats, x)) -> caption first.
  - backend/attention.py     : attention_function accepts an additive float
                               [seq,seq]/[b,seq,seq] mask straight into SDPA.
  - sampling_function.py:271 : model_function_wrapper(apply_model, {"input","timestep","c","cond_or_uncond"})

Two things are marked VERIFY: they need one real generation to pin down rather
than guess (the exact key names inside `c`, and the internal padding applied to
caption/image token counts). Set plan.debug=True to print them on first run.
"""

import torch

from scripts.zit_core import build_bias

# The fast attention backends (sage / flash) ignore an additive attn_mask, so our
# regional bias would be silently dropped. JointAttention resolves
# `attention_function` as a module global at call time, so we swap that single
# reference to the mask-honoring SDPA path while active, and restore it on teardown.
try:
    from backend.nn import lumina as _lumina
    from backend.attention import attention_pytorch as _mask_safe_attention
except Exception:  # pragma: no cover - only importable inside Forge
    _lumina = None
    _mask_safe_attention = None

# class names that mean "this is the NextDiT family we know how to patch"
DIFFUSION_CLASSES = {"NextDiT"}
ATTENTION_CLASSES = {"JointAttention"}


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def get_diffusion_model(p):
    try:
        return p.sd_model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return None


def is_supported(p) -> bool:
    dm = get_diffusion_model(p)
    return dm is not None and dm.__class__.__name__ in DIFFUSION_CLASSES


# --------------------------------------------------------------------------- #
# caption encoding: one prompt -> embeddings, concatenate, record spans
# --------------------------------------------------------------------------- #
def encode_caption(p, text: str):
    """Return the caption embedding tensor for a single prompt, shape [1, L, D].

    get_learned_conditioning on the ZImage engine returns the Qwen3 caption
    features. VERIFY: whether it returns a bare tensor or a (tensor, mask)
    container — printed when plan.debug is on.
    """
    def _to_3d(t):
        # caption tensors come back as [tokens, dim]; normalize to [1, tokens, dim]
        return t.unsqueeze(0) if t.dim() == 2 else t

    cond = p.sd_model.get_learned_conditioning([text])
    # ZImage engine returns a list of per-chunk tensors; normalize + concat on
    # the token axis.
    if isinstance(cond, (list, tuple)):
        tensors = [_to_3d(c) for c in cond if isinstance(c, torch.Tensor)]
        if not tensors:
            raise RuntimeError(f"[zit] no tensors in conditioning list: {cond!r}")
        return torch.cat(tensors, dim=1)
    if isinstance(cond, torch.Tensor):
        return _to_3d(cond)
    if isinstance(cond, dict):
        for k in ("crossattn", "c_crossattn", "cond"):
            if k in cond and isinstance(cond[k], torch.Tensor):
                return _to_3d(cond[k])
    raise RuntimeError(
        f"[zit] unexpected conditioning type {type(cond)}; inspect and extend encode_caption()"
    )


def build_caption_stack(p, plan):
    """Encode base + each region, concat along token axis, fill in spans.

    Returns the concatenated caption tensor [1, L_total, D] and sets
    plan.base_span / region.span (column ranges into that tensor).
    """
    parts, col = [], 0

    if plan.base_prompt:
        emb = encode_caption(p, plan.base_prompt)
        L = emb.shape[1]
        plan.base_span = (col, col + L)
        parts.append(emb)
        col += L

    for r in plan.regions:
        emb = encode_caption(p, r.prompt)
        L = emb.shape[1]
        r.span = (col, col + L)
        parts.append(emb)
        col += L

    stack = torch.cat(parts, dim=1)
    plan.cap_raw_len = stack.shape[1]
    plan._split_logged = False
    plan._bias_logged = False
    if plan.debug:
        print(f"[zit] caption stack: {stack.shape}, base_span={plan.base_span}, "
              f"region_spans={[r.span for r in plan.regions]}")
    return stack


# --------------------------------------------------------------------------- #
# monkeypatch JointAttention.forward
# --------------------------------------------------------------------------- #
def install(p, plan):
    dm = get_diffusion_model(p)
    plan.pad_mult = getattr(dm, "pad_tokens_multiple", None)
    patched = []
    for name, module in dm.named_modules():
        if module.__class__.__name__ in ATTENTION_CLASSES:
            module._zit_orig_forward = module.forward
            module.forward = _make_forward(plan, module)
            patched.append(name)
    plan._patched_modules = (dm, patched)

    # force the mask-honoring attention path while active
    plan._orig_attn_fn = None
    if _lumina is not None and _mask_safe_attention is not None:
        plan._orig_attn_fn = _lumina.attention_function
        _lumina.attention_function = _mask_safe_attention

    if plan.debug:
        was = getattr(plan._orig_attn_fn, "__name__", None)
        print(f"[zit] patched {len(patched)} JointAttention modules, "
              f"pad_tokens_multiple={plan.pad_mult}; attention backend was "
              f"{was}, forcing mask-safe attention_pytorch while active")


def remove(plan):
    info = getattr(plan, "_patched_modules", None)
    if not info:
        return
    dm, names = info
    lookup = dict(dm.named_modules())
    for name in names:
        m = lookup.get(name)
        if m is not None and hasattr(m, "_zit_orig_forward"):
            m.forward = m._zit_orig_forward
            del m._zit_orig_forward

    if getattr(plan, "_orig_attn_fn", None) is not None and _lumina is not None:
        _lumina.attention_function = plan._orig_attn_fn
        plan._orig_attn_fn = None


def _roundup(n, mult):
    if not mult:
        return n
    return n + ((-n) % mult)


def _log_once(plan, attr, msg):
    if plan.debug and not getattr(plan, attr, False):
        print(msg)
        setattr(plan, attr, True)


def _make_forward(plan, module):
    orig = module._zit_orig_forward

    def forward(x, x_mask=None, freqs_cis=None, transformer_options={}):
        if plan.active and plan.grid is not None:
            seq = x.shape[1]
            img_real = plan.grid[0] * plan.grid[1]
            cap_len = _roundup(plan.cap_raw_len, plan.pad_mult)
            img_pad = _roundup(img_real, plan.pad_mult)
            if plan.debug and not plan._split_logged:
                print(f"[zit] split: seq={seq} cap_len={cap_len} "
                      f"img_real={img_real} img_pad={img_pad} "
                      f"(sum={cap_len + img_pad})")
                plan._split_logged = True

            if seq == cap_len + img_pad:
                # main layers: full joint sequence. Cross-region image<->image
                # isolation is only kept for the early steps (composition), then
                # released so the scene fuses; text routing stays on throughout.
                image_self = getattr(plan, "image_self_active", True)
                strength = getattr(plan, "cur_strength",
                                   getattr(plan, "separation_strength", float("inf")))
                bias = _get_bias(plan, cap_len, img_real, img_pad, x.dtype, x.device,
                                 image_self=image_self, strength=strength)
                x_mask = bias if x_mask is None else _combine(x_mask, bias)
                _log_once(plan, "_bias_logged", f"[zit] full bias APPLIED on seq={seq} "
                          f"(base_span={plan.base_span}, spans={[r.span for r in plan.regions]})")
            elif seq == cap_len:
                # context_refiner: caption-only, keep regions from blending
                bias = _get_bias(plan, cap_len, img_real, 0, x.dtype, x.device,
                                 image_self=False)
                x_mask = bias if x_mask is None else _combine(x_mask, bias)
                _log_once(plan, "_capmask_logged", f"[zit] caption-self mask APPLIED on seq={seq}")
            else:
                # noise_refiner (image-only) or anything unexpected: no caption
                # involved, nothing to route.
                _log_once(plan, "_skip_logged", f"[zit] no mask on seq={seq} (image-only / unmatched)")
        return orig(x, x_mask, freqs_cis, transformer_options)

    return forward


def _get_bias(plan, cap_len, img_real, img_pad, dtype, device, image_self=True,
              strength=float("inf")):
    key = (cap_len, img_real, img_pad, image_self, strength, str(device), str(dtype))
    if key not in plan._bias_cache:
        plan._bias_cache[key] = build_bias(plan, cap_len, img_real, img_pad,
                                           device, dtype, image_self=image_self,
                                           strength=strength)
    return plan._bias_cache[key]


def _combine(existing_mask, bias):
    """Add our additive bias onto whatever mask the model already produced.
    Bool masks (True=keep) are converted to additive first."""
    if existing_mask.dtype == torch.bool:
        add = torch.zeros_like(bias)
        add.masked_fill_(~existing_mask, float("-inf"))
        return add + bias
    return existing_mask + bias
