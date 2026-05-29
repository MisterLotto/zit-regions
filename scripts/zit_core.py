"""
zit-regions: architecture-independent core.

Three responsibilities, none of them touch the model:
  1. parse the region grammar (ratios + BREAK-separated prompts)
  2. rasterize regions into per-region masks on the latent *token* grid
  3. build the additive [seq, seq] attention bias that does the actual regional
     restriction inside joint attention

The model-specific glue (detection, monkeypatching, caption encoding, the
text/image split point) lives in an adapter module, e.g. scripts/zit_zimage.py.
"""

import re
from dataclasses import dataclass, field

import torch

# extra-network / lora tags like <lora:name:1>, <hypernet:...>; Forge harvests
# these from the main prompt itself, so strip them from the per-region text.
_TAG_RE = re.compile(r"<[^>]*>")

NEG_INF = float("-inf")


@dataclass
class Region:
    prompt: str
    # fractional bounding box on the token grid, (x0, y0, x1, y1) in [0, 1].
    box: tuple = (0.0, 0.0, 1.0, 1.0)
    # filled in after encoding: [start, end) column span in the caption axis.
    span: tuple = None


@dataclass
class Plan:
    """Everything the hook needs for one generation. Built in process(), read
    inside the patched attention forward."""

    base_prompt: str                 # common prompt every region attends to
    regions: list                    # list[Region]
    mode: str = "columns"            # "columns" | "rows"
    base_span: tuple = None          # column span of the base prompt
    grid: tuple = None               # (gh, gw) token grid, set at gen time
    mask_image_self: bool = True     # block cross-region image<->image attention
    overlap: float = 0.0             # fraction each region extends into neighbors
    debug: bool = False
    # caches, set at runtime:
    _bias_cache: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. grammar
# --------------------------------------------------------------------------- #
def parse_regions(prompt: str, ratios: str, mode: str) -> Plan:
    """
    prompt: chunks separated by 'BREAK'. First chunk = base/common prompt,
            remaining chunks = one per region.
    ratios: e.g. "1,2,1" -> relative sizes of the regions along `mode`.
    mode:   "columns" (split left->right) or "rows" (top->bottom).
    """
    chunks = [_TAG_RE.sub("", c).strip() for c in prompt.split("BREAK")]
    chunks = [c for c in chunks if c]
    if not chunks:
        return Plan(base_prompt="", regions=[])

    base, region_prompts = chunks[0], chunks[1:]

    if not region_prompts:
        return Plan(base_prompt=base, regions=[], mode=mode)

    try:
        weights = [float(x) for x in ratios.replace(" ", "").split(",") if x]
    except ValueError:
        weights = []
    if len(weights) != len(region_prompts):
        weights = [1.0] * len(region_prompts)  # equal split fallback

    total = sum(weights)
    regions, acc = [], 0.0
    for w, rp in zip(weights, region_prompts):
        lo, hi = acc / total, (acc + w) / total
        acc += w
        if mode == "rows":
            box = (0.0, lo, 1.0, hi)
        else:  # columns
            box = (lo, 0.0, hi, 1.0)
        regions.append(Region(prompt=rp, box=box))

    return Plan(base_prompt=base, regions=regions, mode=mode)


# --------------------------------------------------------------------------- #
# 2. rasterize boxes -> flat token masks
# --------------------------------------------------------------------------- #
def region_token_masks(plan: Plan, device, dtype, overlap: float = 0.0) -> list:
    """Return list of flat bool tensors, one per region, length gh*gw.

    `overlap` extends each region into its neighbors along the split axis. Used
    only for image<->image bridging; image->text routing uses sharp (overlap=0)
    masks so each patch reads exactly one region's prompt."""
    gh, gw = plan.grid
    masks = []
    for r in plan.regions:
        x0, y0, x1, y1 = r.box
        if overlap > 0.0:
            if plan.mode == "rows":
                y0, y1 = max(0.0, y0 - overlap), min(1.0, y1 + overlap)
            else:
                x0, x1 = max(0.0, x0 - overlap), min(1.0, x1 + overlap)
        m = torch.zeros((gh, gw), dtype=torch.bool, device=device)
        cx0, cx1 = int(round(x0 * gw)), int(round(x1 * gw))
        cy0, cy1 = int(round(y0 * gh)), int(round(y1 * gh))
        cx1 = max(cx1, cx0 + 1)
        cy1 = max(cy1, cy0 + 1)
        m[cy0:cy1, cx0:cx1] = True
        masks.append(m.reshape(-1))
    return masks


# --------------------------------------------------------------------------- #
# 3. the additive attention bias
# --------------------------------------------------------------------------- #
def _caption_self_block(plan: Plan, cap_len: int, device, dtype) -> torch.Tensor:
    """[cap_len, cap_len] additive mask keeping each region's caption tokens
    from attending across regions. Without this the shared context_refiner (and
    the caption<->caption corner of the main layers) blends every region's text
    together, which collapses the regions into one merged subject.

    Each region's tokens see themselves + the base prompt; base tokens see only
    the base. Padding rows (beyond the real caption length) are left open so no
    row is fully -inf."""
    cc = torch.full((cap_len, cap_len), NEG_INF, dtype=dtype, device=device)

    base = plan.base_span
    if base is not None:
        bl, bh = base
        cc[bl:bh, bl:bh] = 0.0  # base attends base

    for r in plan.regions:
        if r.span is None:
            continue
        lo, hi = r.span
        cc[lo:hi, lo:hi] = 0.0          # region attends itself
        if base is not None:
            cc[lo:hi, bl:bh] = 0.0      # ... and the base prompt

    # caption padding tokens (cap_pad): keep their rows non-degenerate.
    raw = getattr(plan, "cap_raw_len", cap_len)
    if raw < cap_len:
        cc[raw:cap_len, :] = 0.0
    return cc


def build_bias(plan: Plan, cap_len: int, img_real: int, img_pad: int,
               device, dtype, image_self: bool = True,
               strength: float = float("inf")) -> torch.Tensor:
    """
    Joint sequence is [caption tokens | image tokens] (caption first), caption
    padded to `cap_len`, image to `img_pad` (>= img_real real patches).

    When img_pad == 0 this builds the caption-only mask for the context_refiner
    (seq == cap_len). Otherwise it builds the full [seq, seq] mask: the same
    caption-self block in the top-left corner, plus the image->caption routing
    (each image patch reads only the base columns + the columns of the
    region(s) it falls in). image<->image and caption->image are left open.

    Additive float mask (0 = allow, -inf = block), ready for attention_function.
    """
    cc = _caption_self_block(plan, cap_len, device, dtype)
    if img_pad == 0:
        return cc

    seq = cap_len + img_pad
    bias = torch.zeros((seq, seq), dtype=dtype, device=device)
    bias[:cap_len, :cap_len] = cc

    # image rows attend to caption cols: start fully blocked, then open up.
    img_to_cap = torch.full((img_pad, cap_len), NEG_INF, dtype=dtype, device=device)
    if plan.base_span is not None:
        lo, hi = plan.base_span
        img_to_cap[:, lo:hi] = 0.0

    # SHARP masks for text routing: each patch reads exactly one region's prompt
    sharp = region_token_masks(plan, device, dtype, overlap=0.0)

    sharp_padded = []
    sharp_assigned = torch.zeros(img_pad, dtype=torch.bool, device=device)
    for rows in sharp:
        pr = torch.zeros(img_pad, dtype=torch.bool, device=device)
        pr[:img_real] = rows
        sharp_padded.append(pr)
        sharp_assigned |= pr

    for r, rows in zip(plan.regions, sharp_padded):
        if r.span is None:
            continue
        lo, hi = r.span
        img_to_cap[rows, lo:hi] = 0.0

    if plan.base_span is None:
        # no base prompt: patches in no region must see something
        img_to_cap[~sharp_assigned, :] = 0.0

    bias[cap_len:, :cap_len] = img_to_cap

    # image<->image: block cross-region attention so content can't bleed
    # spatially. Patches only attend within their own region; patches in no
    # region (incl. padding) attend everywhere so they don't NaN and can carry
    # shared background.
    # image<->image: penalize (or block) cross-region attention. A finite
    # `strength` softly discourages bleed while keeping the scene shared; inf is
    # a hard cut (renders as separate images). Within-region pairs and any
    # unassigned/padding patches are left at 0.
    if image_self and strength > 0:
        # OVERLAPPED masks for bridging: patches in the shared band belong to
        # both regions, so the two halves stay connected into one scene.
        bridge = region_token_masks(plan, device, dtype, overlap=getattr(plan, "overlap", 0.0))
        bridge_padded = []
        bridge_assigned = torch.zeros(img_pad, dtype=torch.bool, device=device)
        for rows in bridge:
            pr = torch.zeros(img_pad, dtype=torch.bool, device=device)
            pr[:img_real] = rows
            bridge_padded.append(pr)
            bridge_assigned |= pr

        penalty = -strength
        ii = torch.full((img_pad, img_pad), penalty, dtype=dtype, device=device)
        for pr in bridge_padded:
            blk = pr.unsqueeze(1) & pr.unsqueeze(0)
            ii[blk] = 0.0
        unassigned = ~bridge_assigned
        ii[unassigned, :] = 0.0
        ii[:, unassigned] = 0.0
        bias[cap_len:, cap_len:] = ii

    return bias
