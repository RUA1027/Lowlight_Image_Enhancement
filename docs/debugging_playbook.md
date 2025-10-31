# Debugging Playbook: SID NewBP + NAFNet

This guide captures the reset requested by your advisor: stop large-scale training,
prove each component works on the smallest possible problem, then scale up in controlled steps.

## What Changed in the Repository
- Added `tools/debug_overfit.py`, a single-batch overfit harness that instantiates a tiny NewBP-NAFNet and logs loss plus gradient norms in seconds.
- Rigged `HybridLossPlus` to crash fast when any loss term returns `NaN` or `Inf`, pointing to the exact component instead of letting training silently diverge.
- Added `configs/debug/sid_newbp_mono_debug.yml`, a tiny BasicSR config (1× batch, `128×128` patches, plain L1) intended for quick end-to-end smoke tests once the single batch passes.

Keep these three additions as your base camp; every scale-up starts from here.

## Phase 0 – Environment Check
1. Make sure the Python environment activates correctly (the Colab log shows CUDA + PyTorch 2.8 worked). Fix local shell profiles (`conda.exe` missing) before running anything.
2. Run `pip install -r requirements.txt` (or reuse the Colab environment) to guarantee `torch`, `rawpy`, `kornia`, etc., are present before the debug steps.

## Phase 1 – Synthetic Single Batch
Goal: prove the core model + optimizer + backprop are numerically sound.

```bash
python tools/debug_overfit.py --device cuda --iters 200 --log-interval 20
```

- Expected behaviour: loss should drop rapidly (to ~1e-3 or lower) and gradient norm stays finite.
- If you only have CPU locally, add `--device cpu` (it will be slower but still finish quickly).
- Any exception now tells you exactly which loss term misbehaved.

Only after this passes should you touch real data.

## Phase 2 – One Real Pair (Tiny Dataset)
1. Pick a single short/long pair from the SID dataset and build a miniature LMDB/manifest (copy the existing JSON, keep only one `pair_id`, regenerate LMDB with just that pair).
2. Point `configs/debug/sid_newbp_mono_debug.yml` to those debug assets.
3. Run:
   ```bash
   cd NAFNet_base
   python basicsr/train.py -opt ../configs/debug/sid_newbp_mono_debug.yml
   ```
4. Expect loss to go down over a few hundred iters. If it stalls:
   - inspect the generated crops (export with a quick inference script),
   - verify exposure ratios and alignment routines,
   - check that data falls in `[0,1]`.

## Phase 3 – Reintroduce Physics & Color Terms
1. Toggle `--loss hybrid --enable-phys` in `tools/debug_overfit.py` first, still on synthetic data. This downloads VGG once but lets you verify the physics term in isolation.
2. When introducing to real data, enable `use_phys` in the debug config and leave `DeltaE`/`LPIPS` disabled until physics is stable.
3. If the crash reports `HybridLossPlus detected non-finite values in term 'Phys_raw'` (or similar), dump the intermediate tensors around that term to locate bad ratios or division.

## Phase 4 – Gradually Scale Up
Once the debug config produces sensible losses on a *tiny* subset:
1. Increase `samples_per_pair` and `patch_size` incrementally (e.g., 128 → 192 → 256).
2. Re-enable perceptual/DeltaE one switch at a time, watching the NaN guard.
3. Only when everything remains stable on ~50–100 pairs should you attempt the full Colab configuration again.

## Phase 5 – Full Training Readiness Checklist
- `tools/debug_overfit.py` passes with `--loss hybrid --enable-phys --iters 400`.
- `configs/debug/sid_newbp_mono_debug.yml` trains without NaNs and produces visually improved outputs relative to inputs.
- Validation metrics behave: PSNR/SSIM improve from iteration 0.
- Unit tests (`pytest core_tests`) still pass.

If any box above is unchecked, do **not** launch the 300k-iter job; revisit the failing phase.

## Reading the New NaN Messages
When the BasicSR training loop now dies with something like:

```
RuntimeError: HybridLossPlus detected non-finite values in term 'Phys_raw'.
```

interpret it as:
- `Phys_raw`: RAW-domain physics term exploded—inspect exposure ratios and PSF inputs.
- `Perc` / `DeltaE` / `LPIPS`: look for out-of-range sRGB values (not clamped to `[0,1]`).
- `L1_raw`: model outputs diverged entirely (check optimizer, learning rate).

Fix the specific source, rerun the single-batch harness, *then* return to end-to-end training.

## Suggested Weekly Routine
- Start every new idea by duplicating the debug config or script; change a single knob.
- Track loss/grad curves (export from the script or tensorboard) and annotate what changed.
- Never jump two phases at once; if Phase 2 fails after a change, go back to Phase 1 with that change applied.

Following this ladder keeps experiments cheap, reproducible, and aligned with your advisor’s expectation of fast, skeptical iteration.
