import types
import numpy as np
import torch
import argparse
import glob
import os

from src.depth_anything_3.api import DepthAnything3
from src.CUT3R.src.dust3r.model import ARCroco3DStereo
from src.CUT3R.src.dust3r.inference import inference
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
    return parser.parse_args()


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
    return model


def run_inference_training_matched(model, image_paths_batch, args):
    views = _build_views_from_image_paths(image_paths_batch, size=args.input_size)
    outputs, state_args = inference(views, model, args.device, verbose=True)
    return outputs, state_args


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

    # Perform online inference over pairs, save, and aggregate in one pass.
    all_views, all_preds = [], []
    last_state_args = None
    start_idx = 0

    # for batch in pairs:
    #     outputs, state_args = run_inference_training_matched(model, batch, args)

    #     # save without overwriting previous pair files
    #     start_idx = save_geometry_outputs_cuteanything(outputs, args.output_path, start_idx=start_idx)

    # after model = setup_models_and_patch(args)

    image_paths = sorted(glob.glob(os.path.join(args.input_path, "*.png")))
    pairs = [image_paths[i:i+args.views_per_step] for i in range(0, len(image_paths)-1, args.views_per_step)]
    pairs = [p for p in pairs if len(p) == args.views_per_step]
    # flatten in temporal order
    seq_paths = [p for pair in pairs for p in pair]

    # build all views once
    views = _build_views_from_image_paths(seq_paths, size=args.input_size)

    # optional but good hygiene for single continuous sequence
    for i, v in enumerate(views):
        v["idx"] = i
        v["instance"] = f"seq_{i:06d}"
        v["update"] = torch.tensor(True).unsqueeze(0)
        v["reset"] = torch.tensor(i == 0).unsqueeze(0)   # first frame reset

    # ONE call only
    outputs, state_args = inference(views, model, args.device, verbose=True)

    # save once
    save_geometry_outputs_cuteanything(outputs, args.output_path, start_idx=0)
