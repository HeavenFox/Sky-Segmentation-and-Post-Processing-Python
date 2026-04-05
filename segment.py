"""Sky segmentation with U2NetP (PyTorch) and optional mask refinement."""

import os
import argparse

import cv2 as cv
import numpy as np
import torch

from model import U2NetP
from postprocess import refine_mask_with_bilateral


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def load_model(weights_path, device='cpu'):
    model = U2NetP()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image, input_size=384):
    """Preprocess BGR image for model input.

    Matches NCNN inference: downscale large images, resize to input_size,
    BGR→RGB, ImageNet normalize.
    """
    dst = image.copy()
    while dst.shape[0] > 768 and dst.shape[1] > 768:
        dst = cv.pyrDown(dst)
    rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB).astype(np.float32)
    resized = cv.resize(rgb, (input_size, input_size))
    normalized = (resized - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def run_inference(model, image, input_size=384, device='cpu'):
    """Run sky segmentation inference.

    Returns:
        float32 HxW mask in [0, 1] at model input resolution
    """
    x = preprocess(image, input_size).to(device)
    with torch.no_grad():
        fused = model(x)[0]
    return fused.squeeze().cpu().numpy()


def guided_filter(guide, src, radius, eps):
    """Grayscale guided filter for edge-preserving upscaling."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv.boxFilter(guide, -1, ksize)
    mean_p = cv.boxFilter(src, -1, ksize)
    mean_Ip = cv.boxFilter(guide * src, -1, ksize)
    mean_II = cv.boxFilter(guide * guide, -1, ksize)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv.boxFilter(a, -1, ksize)
    mean_b = cv.boxFilter(b, -1, ksize)

    return mean_a * guide + mean_b


def main():
    parser = argparse.ArgumentParser(description='Sky segmentation with U2NetP')
    parser.add_argument('images', nargs='+', help='Input image path(s)')
    parser.add_argument('--weights', default='skyseg_u2netp.pth',
                        help='PyTorch weights path (default: skyseg_u2netp.pth)')
    parser.add_argument('--input-size', type=int, default=384,
                        help='Model input size (default: 384)')
    parser.add_argument('--refine', action='store_true',
                        help='Use advanced mask refinement (from mask_refine.cpp)')
    parser.add_argument('--guided-filter', action='store_true',
                        help='Use guided filter for edge refinement')
    parser.add_argument('--radius', type=int, default=16,
                        help='Guided filter radius (default: 16)')
    parser.add_argument('--eps', type=float, default=1e-3,
                        help='Guided filter epsilon (default: 1e-3)')
    parser.add_argument('--device', default='cpu',
                        help='Device (cpu, cuda, mps)')
    args = parser.parse_args()

    model = load_model(args.weights, args.device)

    for image_path in args.images:
        image = cv.imread(image_path)
        if image is None:
            print(f'Warning: could not read {image_path}, skipping')
            continue
        orig_h, orig_w = image.shape[:2]

        # Inference
        mask = run_inference(model, image, args.input_size, args.device)

        # Upscale and refine
        if args.refine:
            mask_full = cv.resize(mask, (orig_w, orig_h),
                                  interpolation=cv.INTER_LINEAR)
            reference = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
            refined = refine_mask_with_bilateral(mask_full, reference)
            output_mask = (refined * 255).astype(np.uint8)
        elif args.guided_filter:
            mask_up = cv.resize(mask, (orig_w, orig_h),
                                interpolation=cv.INTER_LINEAR)
            guide = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            refined = guided_filter(guide, mask_up, args.radius, args.eps)
            output_mask = np.clip(refined * 255, 0, 255).astype(np.uint8)
        else:
            output_mask = cv.resize(mask, (orig_w, orig_h),
                                    interpolation=cv.INTER_LINEAR)
            output_mask = (output_mask * 255).astype(np.uint8)

        # Build BGRA with alpha (0=sky, 255=not sky)
        alpha = (255 - output_mask).astype(np.uint8)
        bgra = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha

        # Save
        base, _ = os.path.splitext(image_path)
        mask_path = base + '_mask.png'
        result_path = base + '_result.png'

        cv.imwrite(mask_path, output_mask)
        cv.imwrite(result_path, bgra)

        print(f'[{image_path}]')
        print(f'  mask   ({output_mask.shape[1]}x{output_mask.shape[0]}): {mask_path}')
        print(f'  result ({bgra.shape[1]}x{bgra.shape[0]}): {result_path}')


if __name__ == '__main__':
    main()
