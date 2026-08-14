import hashlib
import os
import os.path as osp
import pickle
import sys

try:
    import fcntl  # POSIX advisory file locking (Linux/Mac)
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class DL3DV_Multi(BaseMultiViewDataset):
    def __init__(
        self,
        *args,
        split,
        ROOT,
        is_metric=False,
        feed_gt_ray_map=False,
        feed_prev_gt_ray_map=False,
        ray_cond_noise_mode="",
        ray_cond_noise_t_frac=0.0,
        ray_cond_noise_r_deg=0.0,
        ray_cond_noise_clean_frac=0.10,
        ray_cond_noise_m_lo=0.25,
        ray_cond_noise_m_hi=1.5,
        ray_cond_noise_r_m_cap=1.0,
        ray_cond_noise_drift_t=0.0,
        ray_cond_noise_drift_r=0.0,
        **kwargs,
    ):
        assert not (feed_gt_ray_map and feed_prev_gt_ray_map), (
            "feed_gt_ray_map and feed_prev_gt_ray_map are mutually exclusive: "
            "a view is conditioned either on its own GT camera or its predecessor's"
        )
        # NGC noise-conditioned training (campaign 2026-08-14). All keys are
        # default-off: mode "" leaves every code path and RNG stream
        # byte-identical to a tree without these kwargs (falsifier F1).
        assert ray_cond_noise_mode in ("", "walk", "white", "mix")
        if ray_cond_noise_mode:
            assert feed_gt_ray_map or feed_prev_gt_ray_map, (
                "ray_cond_noise_* perturbs the conditioning ray maps; it is "
                "meaningless without a GT-ray conditioning mode")
        self.ray_cond_noise_mode = ray_cond_noise_mode
        self.ray_cond_noise_t_frac = float(ray_cond_noise_t_frac)
        self.ray_cond_noise_r_deg = float(ray_cond_noise_r_deg)
        self.ray_cond_noise_clean_frac = float(ray_cond_noise_clean_frac)
        self.ray_cond_noise_m_lo = float(ray_cond_noise_m_lo)
        self.ray_cond_noise_m_hi = float(ray_cond_noise_m_hi)
        self.ray_cond_noise_r_m_cap = float(ray_cond_noise_r_m_cap)
        self.ray_cond_noise_drift_t = float(ray_cond_noise_drift_t)
        self.ray_cond_noise_drift_r = float(ray_cond_noise_drift_r)
        self.ROOT = ROOT
        self.video = True
        self.max_interval = 20
        # self.max_interval = 1
        self.is_metric = is_metric
        self.feed_gt_ray_map = feed_gt_ray_map
        self.feed_prev_gt_ray_map = feed_prev_gt_ray_map
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    # ------------------------------------------------------------------
    # Two-layer dataset index: a cached "raw scan" + a cheap in-memory derive.
    #
    # Enumerating the dataset is dominated by filesystem-metadata latency: for
    # tens of thousands of scenes whose data lives on a slow/frozen store reached
    # through symlinks, the per-scene directory walk does ~3 listings per scene and
    # can take tens of minutes -- and every DDP rank repeats it.
    #
    # The expensive part (per-scene rgb file lists) depends ONLY on ROOT + the
    # directory contents -- NOT on num_views/allow_repeat. So we split it:
    #   * _build_raw_scan()  -- the slow walk; cached to disk keyed on (ROOT) plus
    #                           a cheap signature of the top-level dirs.
    #   * _derive_index()    -- the cheap part (cut_off / start-frame math); run in
    #                           memory on every load for the current num_views /
    #                           allow_repeat. Sub-second, no disk needed.
    # Consequence: changing num_views/allow_repeat re-derives instantly and does
    # NOT trigger another walk; only changing ROOT or the scene symlink set does.
    #
    # Env knobs:
    #   DL3DV_CACHE_DISABLE=1  -> bypass the disk cache (always re-walk).
    #   DL3DV_CACHE_DIR=<dir>  -> where scan files live (default ~/.cache/cut3r_dl3dv).
    # ------------------------------------------------------------------

    CACHE_VERSION = 2  # bump to invalidate all previously written caches

    def _scan_signature(self):
        """Cheap fingerprint of the dataset directory used to detect changes.

        Only does one listdir of ROOT plus a stat + listdir of each top-level
        (level-1) directory -- NOT the expensive per-scene walk. Adding or
        removing scene symlinks changes the level-1 directory's mtime and entry
        count, which invalidates the cache. This assumes the symlinked scene
        contents themselves (on the frozen store) do not change in place.

        Deliberately does NOT include num_views/allow_repeat: those affect only
        the cheap derive step, not the scanned rgb file lists.
        """
        level1 = []
        for name in sorted(os.listdir(self.ROOT)):
            p = osp.join(self.ROOT, name)
            if not os.path.isdir(p):
                continue
            st = os.stat(p)
            level1.append((name, int(st.st_mtime_ns), len(os.listdir(p))))
        return {
            "version": self.CACHE_VERSION,
            "root": osp.abspath(self.ROOT),
            "level1": level1,
        }

    def _scan_path(self):
        cache_dir = os.environ.get(
            "DL3DV_CACHE_DIR",
            osp.join(osp.expanduser("~"), ".cache", "cut3r_dl3dv"),
        )
        os.makedirs(cache_dir, exist_ok=True)
        # Keyed on ROOT + version only -- intentionally independent of
        # num_views/allow_repeat so different view counts share one scan.
        key = "|".join([osp.abspath(self.ROOT), f"v={self.CACHE_VERSION}"])
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        return osp.join(cache_dir, f"dl3dv_scan_{h}.pkl")

    def _load_cache(self, cache_path, signature):
        if not osp.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
        except Exception as e:  # corrupt/partial/incompatible -> rebuild
            print(f"[DL3DV_Multi] ignoring unreadable cache {cache_path}: {e}")
            return None
        if blob.get("signature") != signature:
            print(f"[DL3DV_Multi] cache stale, rebuilding scan: {cache_path}")
            return None
        return blob["data"]

    def _save_cache(self, cache_path, signature, data):
        # Write to a unique temp file then atomically rename, so concurrent
        # readers never observe a half-written cache.
        tmp = f"{cache_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(
                    {"signature": signature, "data": data},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(tmp, cache_path)
        except Exception as e:
            print(f"[DL3DV_Multi] failed to write cache {cache_path}: {e}")
            if osp.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _load_data(self):
        if os.environ.get("DL3DV_CACHE_DISABLE", "") in ("1", "true", "True"):
            self._apply_index(self._derive_index(self._build_raw_scan()))
            return

        scan_path = self._scan_path()
        signature = self._scan_signature()

        raw_scan = self._load_cache(scan_path, signature)
        if raw_scan is not None:
            print(f"[DL3DV_Multi] loaded raw scan from cache: {scan_path}")
            self._apply_index(self._derive_index(raw_scan))
            return

        # Serialize the expensive walk across DDP ranks: one process acquires the
        # lock and builds + writes the scan while the others block, then they read
        # what it wrote instead of all walking the tree at once.
        if fcntl is not None:
            lock_path = scan_path + ".lock"
            print(
                f"[DL3DV_Multi] no scan cache yet; acquiring build lock "
                f"(one process walks, others wait): {lock_path}",
                flush=True,
            )
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                try:
                    raw_scan = self._load_cache(scan_path, signature)
                    if raw_scan is None:
                        raw_scan = self._build_raw_scan()
                        self._save_cache(scan_path, signature, raw_scan)
                    else:
                        print(f"[DL3DV_Multi] scan built by another process: {scan_path}")
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)
        else:
            raw_scan = self._build_raw_scan()
            self._save_cache(scan_path, signature, raw_scan)

        self._apply_index(self._derive_index(raw_scan))

    @staticmethod
    def _progress(iterable, total=None, desc=""):
        """Wrap an iterable in a tqdm progress bar; fall back to the bare
        iterable if tqdm is unavailable. Used to give live feedback during the
        otherwise-silent (and slow) raw scan."""
        try:
            from tqdm import tqdm

            return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
        except Exception:
            return iterable

    def _apply_index(self, index):
        self.all_scenes = index["all_scenes"]
        self.scenes = index["scenes"]
        self.sceneids = index["sceneids"]
        self.images = index["images"]
        self.start_img_ids = index["start_img_ids"]
        self.scene_img_list = index["scene_img_list"]

    def _build_raw_scan(self):
        """Walk ROOT and collect the per-scene rgb file lists. This is the slow
        part (per-scene directory listings over the frozen store) and is what gets
        cached. Independent of num_views/allow_repeat. Returns a dict consumed by
        _derive_index()."""
        all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        subscenes = []
        for scene in all_scenes:
            # not empty (iteration order preserved to match the original index)
            entries = os.listdir(osp.join(self.ROOT, scene))
            for f in self._progress(entries, desc=f"Scanning {scene}"):
                p = osp.join(self.ROOT, scene, f)
                if os.path.isdir(p) and len(os.listdir(p)) > 0:
                    subscenes.append(osp.join(scene, f))

        scans = []  # list of (scene, rgb_paths) for every non-empty subscene
        for scene in self._progress(subscenes, total=len(subscenes), desc="Scanning scenes"):
            scene_dir = osp.join(self.ROOT, scene, "dense")
            rgb_paths = sorted(
                [f for f in os.listdir(os.path.join(scene_dir, "rgb")) if f.endswith(".png")]
            )
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            scans.append((scene, rgb_paths))

        return {"all_scenes": all_scenes, "scans": scans}

    def _derive_index(self, raw_scan):
        """Cheap, in-memory: turn the cached raw scan into the sampling index for
        the current num_views / allow_repeat. No filesystem access. This is where
        cut_off filtering and start-frame ranges are computed, so changing
        num_views/allow_repeat only re-runs this step (sub-second)."""
        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0

        cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)

        for scene, rgb_paths in raw_scan["scans"]:
            num_imgs = len(rgb_paths)
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        return {
            "all_scenes": raw_scan["all_scenes"],
            "scenes": scenes,
            "sceneids": sceneids,
            "images": images,
            "start_img_ids": start_img_ids,
            "scene_img_list": scene_img_list,
        }

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            block_shuffle=25,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for view_idx in image_idxs:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id], "dense")

            rgb_path = self.images[view_idx]
            basename = rgb_path[:-4]

            rgb_image = imread_cv2(osp.join(scene_dir, "rgb", rgb_path), cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, "depth", basename + ".npy")).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
            sky_mask = (
                cv2.imread(osp.join(scene_dir, "sky_mask", rgb_path), cv2.IMREAD_UNCHANGED) >= 127
            )
            outlier_mask = cv2.imread(
                osp.join(scene_dir, "outlier_mask", rgb_path), cv2.IMREAD_UNCHANGED
            )
            depthmap[sky_mask] = -1.0
            depthmap[outlier_mask >= 127] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            # threshold = (
            #     np.percentile(depthmap[depthmap > 0], 99.5)
            #     if depthmap[depthmap > 0].size > 0
            #     else 0
            # )
            # depthmap[depthmap > threshold] = 0.0

            intrinsics = cam_file["intrinsic"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="dl3dv",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=osp.join(scene_dir, "rgb", rgb_path),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.9, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        if self.feed_gt_ray_map or self.feed_prev_gt_ray_map:
            # GT-pose oracle: expose a ground-truth camera to the model through
            # the pretrained ray-map encoder branch (img_mask stays True, so the
            # image is fed alongside the rays). With feed_gt_ray_map each view
            # carries its own camera; with feed_prev_gt_ray_map the ray maps are
            # shifted one step back in BaseMultiViewDataset.__getitem__ so view v
            # carries view v-1's camera. View 0 keeps ray_mask=False: it defines
            # the reference frame (feed_gt_ray_map: its relative pose is identity;
            # feed_prev_gt_ray_map: it has no predecessor).
            for v in range(1, len(views)):
                views[v]["ray_mask"] = True
        return views
