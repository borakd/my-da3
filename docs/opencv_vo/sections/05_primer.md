# Primer: what the five metrics measure, and what the backbone actually computes

This section exists because the results table is easy to misread. Two of its columns cannot move
at all, one is saturated, and the two that do move measure very different things — a *local* and a
*global* property of the same trajectory. Without that distinction the table looks like a
contradiction: the backbone can be simultaneously better than a model on rotation and 37% worse on
trajectory error.

## 1. The five metrics, exactly as the harness computes them

All five come from `eval_bundle/bin/eval_depth_poses.py` with default arguments. Pose metrics are
computed by `evo`; the defaults that matter are `--align sim3` and `--pose_reduce rmse`.

### Pose: one global fit, then two very different error measures

Before any pose metric is computed, the *whole* predicted trajectory is fitted to ground truth by a
single similarity transform — one rotation, one translation, one scale, chosen to minimise
alignment error over the entire scene (`align=True, correct_scale=True`). This is what makes it
legitimate to score a monocular method that has no idea of metric scale.

**ATE** (absolute trajectory error) = `evo` APE with `PoseRelation.translation_part`:
after that single alignment, take the Euclidean distance between the predicted and the true camera
centre at each frame, and reduce over frames by RMSE.

> ATE asks: *after one best-fit rigid-plus-scale placement, how far is the whole path from the true
> path?* It is a *global* measure. Slow drift and any inconsistency of scale along the sequence
> both show up here, because a single similarity transform cannot undo them.

**RPE-trans** and **RPE-rot** (relative pose error) = `evo` RPE with `delta=1, delta_unit=frames,
all_pairs=True`: for every consecutive pair of frames, compose the predicted relative motion with
the inverse of the true relative motion, and measure what is left over — the translation part in
metres for RPE-trans, the rotation angle in degrees for RPE-rot. Again reduced by RMSE over pairs.

> RPE asks: *how wrong is each individual step?* It is a *local* measure. Errors do not accumulate
> in it: a trajectory that drifts far away but gets every single step right scores well on RPE and
> badly on ATE.

### Depth: per-frame scale alignment, then two pixel statistics

**AbsRel** = mean over valid pixels of `|s·pred − gt| / gt`, where `s` is a per-frame scale fitted
to ground truth (`--cut3r_depth_align scale`). **δ<1.25** ("a1") = the fraction of pixels whose
ratio `max(s·pred/gt, gt/(s·pred))` is below 1.25.

Both are computed from **depth maps**, which the backbone never produces or modifies.

### The per-scene → table reduction

Each scene yields one number per metric (RMSE over that scene's frames for pose; mean over pixels
and frames for depth). The table cell is the plain mean of those per-scene numbers over all 4292
scenes, matching `eval_pipeline/aggregate_results.py`.

## 2. What the OpenCV backbone actually computes

Per scene, in one forward pass over the frames:

1. **Corners** are detected on the current frame and **tracked** into the next by pyramidal
   Lucas–Kanade with a forward-backward consistency check. This yields, for each surviving track,
   a 2D pixel position in every frame it lives in. That is the *only* measurement the method has.
2. **A map is bootstrapped** once the tracks have moved 5 px: an essential matrix between frame 0
   and that frame gives a rotation and a translation *direction*, and triangulating the tracks
   against that motion gives 3D points. The translation is returned as a **unit vector**, so the
   map's overall size is arbitrary — nothing in the images tells a single camera how big the world
   is. Everything downstream inherits that arbitrary unit.
3. **Every subsequent frame is posed by PnP**: take the tracks that carry a 3D map point, and find
   the camera rotation and translation that reproject those points closest to where the tracks
   actually are, robustly with RANSAC.
4. **The map grows** at keyframes by triangulating new tracks against the previous keyframe, and
   shrinks by dropping points that PnP keeps calling outliers.
5. **If PnP fails three frames in a row**, the map is declared dead and a new one is bootstrapped
   from the current frame. The new map gets its **own** arbitrary unit; a scale hand-off tries to
   match it to the old one through tracks that survived, and often cannot.

What it never does: it never optimises the trajectory as a whole (no bundle adjustment over the
scene, no loop closure, no pose graph), it never revisits a pose once written, and it never sees
depth, ground truth, or any learned prior. Its only external input is CUT3R's predicted focal
length for the current frame.

## 3. Putting the two together — why the table looks the way it does

**Why the depth columns cannot move.** The backbone outputs camera poses. The depth directory of
an OpenCV row is a symlink to the paired model's. AbsRel and δ<1.25 are therefore the model's own
numbers, bit-identical on all 4292 scenes. They are printed only so the row is a complete
five-metric entry, and must be read as inherited.

**Why rotation is where a classical method shines.** The rotation between two consecutive frames is
determined by how the whole image moves, and is completely independent of scale — you can recover
it from correspondences alone with no idea how far away anything is. It is also the part of the
motion that optical flow measures most reliably. So a geometric solver does well here, and against
the zero-shot network (whose rotation is worse than predicting no rotation at all) it wins by 24%.

**Why RPE-trans tells you nothing here.** The camera moves 3.7 mm between frames at the median.
A trajectory that never moves at all scores 0.0079 m on this metric; the finetuned model scores
0.0079 m and the backbone 0.0084 m. No method in the table resolves a single frame's translation,
so the column is saturated and every row is compressed into it.

**Why ATE is where the backbone loses, and it is not the solver's fault.** ATE is scored after
*one* global similarity fit — one scale for the entire scene. A monocular map has one arbitrary
scale per map. The backbone rebuilds its map 15 071 times across 4292 scenes, a median of twice per
scene, so a typical trajectory is three segments each with an independent scale. Fitting one scale
to three cannot work, and the leftover mismatch is pure ATE error even when every individual step
was estimated perfectly. This is why the same method can tie the finetuned model on both local
metrics while being 37% worse globally: the local estimates are good, and the way they are stitched
into one trajectory is not.

**Why the zero-shot ATE tie is not a compliment to either arm.** Measured on the same 4292 scenes,
a trajectory that takes the ground-truth first step and then coasts at constant velocity scores
0.1248 m. Zero-shot CUT3R scores 0.1197 m (4.1% better than that) and zero-shot + OpenCV 0.1203 m
(3.6% better). Both sit in a four-percent band above a trivial baseline, so their agreement means
neither is saying much about the global trajectory. The finetuned model, at 0.0759 m, is 39.2%
better than trivial and is the only arm in the table with real headroom on this metric.
