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
   `[seq, seq]` bias** so each image patch may only attend to the caption
   columns of the region(s) it falls in (plus the shared base prompt).

Nothing in the Forge install is modified on disk — all patching is runtime
attribute assignment, reverted in `postprocess`.

## Layout

```
scripts/zit_core.py     grammar + mask rasterization + attention bias  (model-agnostic)
scripts/zit_zimage.py   Z-Image adapter: detect / patch / encode captions
scripts/zit_regions.py  Forge Script: UI + lifecycle
```

Adding another joint-attention DiT later = one new adapter; the core is shared.

## Usage

Enable in the **zit-regions** accordion. First chunk of the prompt box is the
common/base prompt; chunks after each `BREAK` are one region each, in split
order. `Region ratios` sets relative sizes along the split direction.

```
a sunny park BREAK a man in a red coat BREAK a woman in a blue dress
ratios: 1,1   direction: columns
```

## Status — working

Confirmed generating distinct characters in a shared scene on Z-Image Turbo
(9-step). Two key findings drove the design:

- **Text routing alone doesn't separate subjects on a DiT.** Image patches also
  attend to *each other*, so content bleeds spatially regardless of which text
  each patch reads. Cross-region image↔image attention must also be restricted.
- **A hard image↔image cut produces two separate images** if held the whole way.
  On few-step turbo, applying it for only the **first ~20% of steps** plants two
  distinct figures, then releasing it lets the scene fuse into one coherent image.

### Controls

- **Separation strength** — penalty on cross-region image attention (16 = hard
  cut, recommended).
- **Region overlap %** — optional bridging band; leave 0 unless a seam shows.
- **Apply for (% of steps)** — the main knob; ~20 is a good default for turbo.

Working recipe: **strength 16, overlap 0, apply 20%**.

Turbo is guidance-distilled, so there is no negative-prompt / CFG path — regions
are positive-only by design.
