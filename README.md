# zit-regions

Regional prompting for **Z-Image / Z-Image Turbo** (NextDiT / Lumina2-family DiT)
in [SD Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).

Lets you assign different prompts to different regions of one image — the
spiritual successor to Regional Prompter / Latent Couple, rebuilt for joint-
attention diffusion transformers instead of the SD/SDXL UNet.

## How it works

Z-Image is a DiT with **joint self-attention**: caption tokens and image tokens
share one sequence (`[caption | image]`, caption first). There is no cross-
attention to hook. Instead this extension:

1. encodes the base prompt and each region prompt separately, then concatenates
   them into one caption sequence (recording each region's column span);
2. injects that stack as the model's caption context via Forge's
   `set_model_unet_function_wrapper`;
3. monkeypatches every `JointAttention.forward` to add an **additive
   `[seq, seq]` bias** that does three things:
   - **caption isolation** — each region's caption tokens don't attend across
     regions (block-diagonal), so the shared refiner can't blend them;
   - **image→text routing** — each image patch reads only its region's caption
     columns (plus the base), using sharp region masks;
   - **image↔image separation** — cross-region image attention is penalized
     (step-gated: hard early, gentle residual after) so subjects don't bleed
     spatially yet the scene still fuses.

Because the fast attention backends (sage/flash) ignore an additive mask, the
adapter temporarily swaps Forge's `attention_function` to the mask-honoring SDPA
path while active.

Nothing in the Forge install is modified on disk — all patching is runtime
attribute assignment, reverted in `postprocess` (and self-reverted if install
fails, so it can't leak into other models).

## Layout

```
scripts/zit_core.py     grammar + mask rasterization + attention bias  (model-agnostic)
scripts/zit_zimage.py   Z-Image adapter: detect / patch / encode captions
scripts/zit_regions.py  Forge Script: UI + lifecycle
```

Adding another joint-attention DiT later = one new adapter; the core is shared.

## Usage

Tick **Enable** in the **zit-regions** accordion, then prompt in the **main
prompt box** using `BREAK`:

- the **first chunk** is the common/base prompt (shared by the whole image —
  setting, lighting, framing; keep people/count words out of it);
- each chunk **after a `BREAK`** is one region, in split order.

`Region ratios` sets relative sizes along the split direction (`Split
direction` = columns or rows).

```
dim cocktail bar, warm tungsten light, wide cinematic shot
BREAK a man in a red suit on the left
BREAK a woman in a black leather dress on the right
ratios: 1,1   direction: columns
```

`<lora:...>` tags work as usual in the main prompt (Forge applies them
globally); they're stripped from the per-region text so they don't pollute the
embeddings. Per-region LoRA *confinement* is not yet implemented — a LoRA
affects the whole image, localized only by the region prompt (same as RP).

Settings are written to the image metadata (`ZIT mode/ratios/strength/overlap/
hardcut/residual`) and restored by **Send to txt2img** / paste-params.

## Status — working

Confirmed generating distinct characters in a shared scene on Z-Image Turbo
(9-step). Two findings drove the design:

- **Text routing alone doesn't separate subjects on a DiT.** Image patches also
  attend to *each other*, so content bleeds spatially regardless of which text
  each patch reads. Cross-region image↔image attention must also be restricted.
- **A constant hard cut produces two separate images; a full release lets them
  re-merge.** The fix is a hybrid: a hard cut over the early steps to commit two
  distinct figures, then a gentle *residual* penalty for the rest so they hold
  without splitting, while the scene fuses.

### Controls

- **Separation strength** — penalty on cross-region image attention during the
  hard-cut window (16 = hard cut).
- **Region overlap %** — optional bridging band; leave 0 unless a seam shows.
- **Hard cut for (% of steps)** — how long the hard cut holds before dropping to
  the residual.
- **Residual separation** — gentle penalty kept for the remaining steps so
  subjects don't re-merge (0 = full release; ~0.5–3 is the useful range).

Starting recipe: **strength 16, overlap 0, hard cut 20–30%, residual 0.5–1**.
Separation is somewhat seed-dependent; tune the residual and lean on a strong
base prompt to fuse the scene.

Turbo is guidance-distilled, so there is no negative-prompt / CFG path — regions
are positive-only by design.
