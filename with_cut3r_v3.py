import types
import numpy as np
import torch
import argparse
import glob
import os

from src.depth_anything_3.api import DepthAnything3
from src.CUT3R.src.dust3r.model import ARCroco3DStereo
from src.CUT3R.src.dust3r.inference import inference
from src.CUT3R.src.dust3r.utils.device import to_cpu
from src.CUT3R.src.dust3r.utils.image import load_images, load_images_da3
from src.CUT3R.viser_utils import PointCloudViewer
from src.CUT3R.demo import prepare_output
from with_cut3r_v1 import get_cut3r_encoder_outputs_from_da3
from with_cut3r_v1 import save_geometry_outputs_cuteanything



def parse_args():
    parser = argparse.ArgumentParser(description="Run DA3 tokens through CUT3R memory.")
    parser.add_argument("--input_path", type=str, required=True, help="Input image directory.")
    parser.add_argument("--output_path", type=str, default="outputs/with_cut3r", help="Output directory.")
    parser.add_argument("--da3_model", type=str, default="depth-anything/DA3-GIANT-1.1", help="DA3 model id/path for DepthAnything3.from_pretrained().")
    parser.add_argument("--cut3r_model", type=str, default="src/CUT3R/src/cut3r_512_dpt_4_64.pth", help="CUT3R checkpoint path or HuggingFace model id.")
    parser.add_argument("--da3_size", type=int, default=504, help="DA3 process resolution.")
    parser.add_argument("--input_size", type=int, default=512, help="Input image size.")
    parser.add_argument("--views_per_step", type=int, default=2, help="Number of synchronized views per timestep in the interleaved stream.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--cam_dir", type=str, default=None, help="Directory of DL3DV_Multi cam/<basename>.npz ground-truth poses. Defaults to the sibling 'cam' dir of --input_path.")
    parser.add_argument("--disable_gt_pose", action="store_true", help="Do not feed ground-truth camera poses into the memory, even if a gt_pose_input model / cam files are available.")
    return parser.parse_args()


def _load_c2w_for_image_path(img_path, cam_dir=None):
    """Ground-truth cam2world (4x4) for an rgb frame, from the sibling DL3DV_Multi
    `cam/<basename>.npz` (key 'pose'). Returns None if unavailable/invalid, so the
    model's guard trips and it degrades to baseline instead of seeing a bogus pose."""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    if cam_dir is None:
        cam_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "cam")
    cam_path = os.path.join(cam_dir, basename + ".npz")
    if not os.path.exists(cam_path):
        return None
    try:
        pose = np.load(cam_path)["pose"].astype(np.float32)  # (4,4) cam2world
    except Exception:
        return None
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return torch.from_numpy(pose)


def _build_views_from_image_paths(image_paths, size=512):
    """
    Build CUT3R-style views (same structure used by demo/inference path).
    """
    images = load_images_da3(image_paths, size=size, square_ok=True, ps=14)
    views = []
    for i, im in enumerate(images):
        img = im["img"]  # [1,3,H,W] normalized
        H, W = img.shape[-2], img.shape[-1]
        view = {
            "img": img,
            "ray_map": torch.full((img.shape[0], 6, H, W), torch.nan),
            "true_shape": torch.from_numpy(im["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views

def setup_models_and_patch(args):
    # Load frozen models.
    model = ARCroco3DStereo.from_pretrained(args.cut3r_model).to(args.device).eval()
    da3_model = DepthAnything3.from_pretrained(args.da3_model).to(args.device).eval()
    for p in da3_model.parameters():
        p.requires_grad = False

    def _encode_views_da3(self, views, img_mask=None, ray_mask=None):
        shape_per_view, feat_per_view, pos_per_view = get_cut3r_encoder_outputs_from_da3(
            batch=views,
            da3_model=da3_model,
            feat_layers=[39],
            da3_size=args.da3_size,
            device=args.device,
        )
        return tuple(shape_per_view), [tuple(feat_per_view)], tuple(pos_per_view)

    model._encode_views = types.MethodType(_encode_views_da3, model)
    model.views_per_step = int(getattr(args, "views_per_step", 1))
    return model


def run_inference_training_matched(model, image_paths_batch, args):
    views = _build_views_from_image_paths(image_paths_batch, size=args.input_size)
    outputs, state_args = inference(views, model, args.device, verbose=True)
    return outputs, state_args


def _move_views_to_device(views, device):
    ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
    for view in views:
        for name in view.keys():
            if name in ignore_keys:
                continue
            if isinstance(view[name], (tuple, list)):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)
    return views


@torch.no_grad()
def run_streamed_inference_and_save(model, seq_paths, args):
    if len(seq_paths) == 0:
        raise ValueError("No sequence views available for inference")

    chunk_size = max(1, int(args.views_per_step))
    total_views = len(seq_paths)

    # Keep only the per-view flags needed by CUT3R decoder state updates, plus the
    # ground-truth camera pose (absolute world-frame cam2world) so a gt_pose_input
    # model can fold it into the recurrent memory. views[0] is global frame 0, so the
    # model's internal inv(views[0]) @ views[i] anchors on the true first frame.
    use_gt_pose = not getattr(args, "disable_gt_pose", False)
    cam_dir = getattr(args, "cam_dir", None)
    global_view_meta = []
    n_with_pose = 0
    for i in range(total_views):
        meta = {
            "img_mask": torch.tensor(True, device=args.device).unsqueeze(0),
            "update": torch.tensor(True, device=args.device).unsqueeze(0),
            "reset": torch.tensor(i == 0, device=args.device).unsqueeze(0),
        }
        if use_gt_pose:
            c2w = _load_c2w_for_image_path(seq_paths[i], cam_dir=cam_dir)
            if c2w is not None:
                meta["camera_pose"] = c2w.unsqueeze(0).to(args.device)  # (1,4,4)
                n_with_pose += 1
        global_view_meta.append(meta)
    if use_gt_pose:
        # Partial coverage is safe: the per-group guard skips GT injection for any
        # step whose views (incl. view 0) lack a finite pose -> never fed identity.
        print(
            f">> GT poses loaded for {n_with_pose}/{total_views} views "
            f"(model.gt_pose_input={getattr(model, 'gt_pose_input', False)}, "
            f"fusion={getattr(model, 'gt_pose_fusion', 'n/a')})"
        )

    state_feat = state_pos = init_state_feat = mem = init_mem = None
    last_state_args = None
    start_idx = 0

    print(f">> Inference with model on {total_views} image/raymaps (chunk={chunk_size})")

    for chunk_start in range(0, total_views, chunk_size):
        chunk_paths = seq_paths[chunk_start : chunk_start + chunk_size]
        chunk_views = _build_views_from_image_paths(chunk_paths, size=args.input_size)

        for local_idx, view in enumerate(chunk_views):
            global_idx = chunk_start + local_idx
            view["idx"] = global_idx
            view["instance"] = f"seq_{global_idx:06d}"
            view["update"] = torch.tensor(True).unsqueeze(0)
            view["reset"] = torch.tensor(global_idx == 0).unsqueeze(0)

        _move_views_to_device(chunk_views, args.device)
        shape, feat_ls, pos = model._encode_views(chunk_views)
        feat = feat_ls[-1]

        if state_feat is None:
            state_feat, state_pos = model._init_state(feat[0], pos[0])
            mem = model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
            init_state_feat = state_feat.clone()
            init_mem = mem.clone()

        chunk_preds = []
        for local_start, local_end in model._group_view_ranges(len(chunk_views)):
            local_indices = list(range(local_start, local_end))
            global_indices = [chunk_start + idx for idx in local_indices]
            res_group, (state_feat, mem) = model._forward_decoder_group_step(
                views=global_view_meta,
                view_indices=global_indices,
                feat_group=[feat[i] for i in local_indices],
                pos_group=[pos[i] for i in local_indices],
                shape_group=[shape[i] for i in local_indices],
                init_state_feat=init_state_feat,
                init_mem=init_mem,
                state_feat=state_feat,
                state_pos=state_pos,
                mem=mem,
            )
            chunk_preds.extend(res_group)

        last_state_args = (state_feat, state_pos, init_state_feat, mem, init_mem)

        chunk_outputs = to_cpu({"views": chunk_views, "pred": chunk_preds})
        start_idx = save_geometry_outputs_cuteanything(
            chunk_outputs, args.output_path, start_idx=start_idx
        )

    return last_state_args


if __name__ == "__main__":
    args = parse_args()
    print(f"Arguments: {args}")

    # Build view pairs
    image_paths = sorted(glob.glob(os.path.join(args.input_path, "*.png")))
    pairs = []
    for i in range(0, len(image_paths) - 1, args.views_per_step):
        pairs.append(image_paths[i:i+args.views_per_step])
    if len(pairs) == 0:
        raise ValueError(f"No valid image pairs found in {args.input_path}")
    print(f"Found {len(pairs)} pairs")

    # Load models once outside of the loop.
    model = setup_models_and_patch(args)

    # Flatten in temporal order and stream in chunks of args.views_per_step.
    seq_paths = [p for pair in pairs for p in pair]
    run_streamed_inference_and_save(model, seq_paths, args)
