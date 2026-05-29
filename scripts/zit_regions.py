"""
zit-regions — regional prompting for Z-Image (NextDiT) in SD Forge.

Front end + lifecycle. The heavy lifting is in:
  scripts/zit_core.py    (grammar, mask raster, attention bias)
  scripts/zit_zimage.py  (model detection, monkeypatch, caption encoding)
"""

import gradio as gr

import modules.scripts as scripts
from modules import shared

from scripts.zit_core import parse_regions
from scripts import zit_zimage as adapter


class ZitRegions(scripts.Script):
    def __init__(self):
        super().__init__()
        self.plan = None

    def title(self):
        return "zit-regions"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("zit-regions", open=False):
            enabled = gr.Checkbox(value=False, label="Enable regional prompting")
            mode = gr.Radio(
                ["columns", "rows"], value="columns", label="Split direction"
            )
            gr.Markdown(
                "Prompt in the **main prompt box**, split with `BREAK`: the first "
                "chunk is the common/base prompt, each chunk after is one region "
                "(in split order).\n\n"
                "`a sunny park BREAK a man in a red coat BREAK a woman in a blue dress`"
            )
            ratios = gr.Textbox(
                value="1,1", label="Region ratios (comma separated)"
            )
            strength = gr.Slider(
                0.0, 16.0, value=16.0, step=0.5,
                label="Separation strength",
                info="Penalty on cross-region image attention. 16 = hard cut "
                     "(recommended, combined with a low 'Apply for %'). Lower "
                     "values soften but rarely help on few-step turbo.",
            )
            overlap = gr.Slider(
                0, 50, value=0, step=5,
                label="Region overlap (%)",
                info="Bridging band between regions (image attention only; text "
                     "routing stays sharp). Usually leave at 0; raise only if a "
                     "hard seam appears down the boundary.",
            )
            isolation = gr.Slider(
                0, 100, value=30, step=5,
                label="Hard cut for (% of steps)",
                info="Hard separation over this fraction of early steps to commit "
                     "two subjects, then drop to the residual below.",
            )
            residual = gr.Slider(
                0.0, 16.0, value=3.0, step=0.5,
                label="Residual separation",
                info="Gentle cross-region penalty kept for the remaining steps so "
                     "subjects don't re-merge while the scene fuses. 0 = full "
                     "release (tends to re-merge); ~2-4 is a good range.",
            )
            debug = gr.Checkbox(value=False, label="Debug (print shapes/spans)")

        # let "send to txt2img" / paste-params restore these from image metadata
        self.infotext_fields = [
            (mode, "ZIT mode"),
            (ratios, "ZIT ratios"),
            (strength, "ZIT strength"),
            (overlap, "ZIT overlap"),
            (isolation, "ZIT hardcut"),
            (residual, "ZIT residual"),
        ]
        self.paste_field_names = [k for _, k in self.infotext_fields]

        return [enabled, mode, ratios, strength, overlap, isolation, residual, debug]

    # ----------------------------------------------------------------- #
    def process_before_every_sampling(self, p, enabled, mode, ratios, strength, overlap, isolation, residual, debug, **kwargs):
        # install here (not process()) so we run AFTER Forge applies LoRAs and
        # rebuilds forge_objects.unet; installing earlier gets wiped by LoRAs.
        self._teardown()  # safety: never leave a stale patch installed

        prompt = self._main_prompt(p)
        if not enabled or "BREAK" not in prompt:
            return
        if not adapter.is_supported(p):
            if debug:
                print("[zit] loaded model is not a supported NextDiT model; skipping")
            return

        plan = parse_regions(prompt, ratios, mode)
        if not plan.regions:
            return
        plan.overlap = max(0.0, min(0.5, overlap / 100.0))
        plan.debug = debug
        plan.active = False  # flipped on once captions + grid are ready

        # cross-region image separation: hard cut early, gentle residual after
        plan.separation_strength = float("inf") if strength >= 16.0 else float(strength)
        plan.residual_strength = float("inf") if residual >= 16.0 else float(residual)
        plan.isolation_frac = max(0.0, min(1.0, isolation / 100.0))
        plan.total_steps = getattr(p, "steps", 0) or 0
        plan.cur_step = 0
        plan.cur_strength = plan.separation_strength
        plan.image_self_active = plan.cur_strength > 0.0

        # token grid: VAE /8 then patch_size 2 -> /16
        plan.grid = (p.height // 16, p.width // 16)

        # encode base + region prompts, concat, record spans
        plan.caption_stack = adapter.build_caption_stack(p, plan)

        # install the attention monkeypatch + the caption-injection wrapper.
        # Register self.plan FIRST so that if install partially completes and
        # then throws, the next run's _teardown() can still revert the global
        # attention swap / monkeypatch instead of leaking into other models.
        self.plan = plan
        try:
            adapter.install(p, plan)
            unet = p.sd_model.forge_objects.unet
            unet.set_model_unet_function_wrapper(self._make_unet_wrapper(plan))
        except Exception:
            self._teardown()
            if debug:
                import traceback
                print("[zit] install failed; reverted. Generation proceeds unmodified.")
                traceback.print_exc()
            return

        plan.active = True

        # record settings into image metadata so a good run can be reproduced
        params = getattr(p, "extra_generation_params", None)
        if params is not None:
            params.update({
                "ZIT mode": mode,
                "ZIT ratios": ratios,
                "ZIT strength": strength,
                "ZIT overlap": overlap,
                "ZIT hardcut": isolation,
                "ZIT residual": residual,
            })

        if debug:
            print(f"[zit] active: grid={plan.grid}, regions={len(plan.regions)}")

    def postprocess(self, p, processed, *args):
        self._teardown()

    # ----------------------------------------------------------------- #
    def _make_unet_wrapper(self, plan):
        """Replace the caption context with our concatenated region stack so the
        joint sequence carries every region's tokens; the attention bias then
        routes each image patch to its region's columns.

        VERIFY on first run: which key in `c` holds the caption context and
        which holds the attention mask (printed when debug is on)."""

        def wrapper(apply_model, params):
            x = params["input"]
            t = params["timestep"]
            c = dict(params["c"])

            if plan.debug:
                shapes = {k: getattr(v, "shape", type(v).__name__) for k, v in c.items()}
                print(f"[zit] unet wrapper c keys -> {shapes}")

            stack = plan.caption_stack.to(device=x.device, dtype=x.dtype)
            stack = stack.expand(x.shape[0], -1, -1)

            # Z-Image: model_conds only carries c_crossattn; no attention_mask
            # (compile_conditions never sets one). Swap in our region stack and
            # let the attention bias do the routing.
            replaced = False
            for key in ("c_crossattn", "crossattn", "context"):
                if key in c:
                    c[key] = stack
                    replaced = True
                    break
            if plan.debug and not replaced:
                print("[zit] WARNING: no caption key found in c; nothing swapped")

            # hard cut over the early window, then drop to the residual penalty
            total = plan.total_steps or 1
            frac = plan.cur_step / max(1, total)
            plan.cur_strength = (plan.separation_strength
                                 if frac < plan.isolation_frac
                                 else plan.residual_strength)
            plan.image_self_active = plan.cur_strength > 0.0
            if plan.debug and plan.cur_step in (0, total - 1):
                print(f"[zit] step {plan.cur_step}/{total} frac={frac:.2f} "
                      f"strength={plan.cur_strength}")
            plan.cur_step += 1

            return apply_model(x, t, **c)

        return wrapper

    @staticmethod
    def _main_prompt(p):
        pr = getattr(p, "prompt", "")
        if isinstance(pr, (list, tuple)):
            pr = pr[0] if pr else ""
        if not pr:
            allp = getattr(p, "all_prompts", None)
            if allp:
                pr = allp[0]
        return pr or ""

    def _teardown(self):
        if self.plan is not None:
            adapter.remove(self.plan)
            self.plan = None
