"""Mask refinement post-processing.

Ported from mask_refine.cpp — implements the confidence-weighted
affine correction algorithm from the paper.
"""

import numpy as np
import cv2 as cv


def bias(x, b=0.8):
    """Bias function (paper formula 2)."""
    return x / (((1.0 / b) - 2.0) * (1.0 - x) + 1.0)


def probability_to_confidence(mask, low_thresh=0.3, high_thresh=0.5):
    """Convert probability mask to confidence map (paper formula 1).

    Args:
        mask: float32 HxW probability map in [0, 1]
        low_thresh: below this → confident non-sky
        high_thresh: above this → confident sky

    Returns:
        confidence: float32 HxW confidence map
    """
    low = mask < low_thresh
    high = mask > high_thresh

    confidence = np.zeros_like(mask)

    # Low confidence: (low_thresh - p) / low_thresh
    if np.any(low):
        conf_low = (low_thresh - mask[low]) / low_thresh
        confidence[low] = bias(conf_low)

    # High confidence: (p - high_thresh) / (1 - high_thresh)
    if np.any(high):
        conf_high = (mask[high] - high_thresh) / (1.0 - high_thresh)
        confidence[high] = bias(conf_high)

    confidence = np.maximum(confidence, 0.01)
    return confidence


def downsample2_antialiased(X):
    """Antialiased 2x downsampling: separable filter + pyrDown."""
    kx = np.array([[1.0 / 8], [3.0 / 8], [3.0 / 8], [1.0 / 8]], dtype=np.float32)
    ky = np.array([[1.0 / 8, 3.0 / 8, 3.0 / 8, 1.0 / 8]], dtype=np.float32)
    dst = cv.sepFilter2D(X, -1, kx, ky, anchor=(1, 1),
                         borderType=cv.BORDER_REPLICATE)
    h, w = dst.shape[:2]
    return cv.pyrDown(dst, dstsize=(round(w / 2), round(h / 2)))


def self_resize(X, target_w, target_h):
    """Iterative antialiased downsampling then linear interpolation to target size."""
    while X.shape[1] >= 2 * target_w and X.shape[0] >= 2 * target_h:
        X = downsample2_antialiased(X)
    return cv.resize(X, (target_w, target_h), interpolation=cv.INTER_LINEAR)


def weighted_downsample(X, confidence, target_w, target_h):
    """Confidence-weighted downsampling.

    Args:
        X: float32 HxW or HxWx3 image
        confidence: float32 HxW confidence map
        target_w, target_h: target dimensions

    Returns:
        Weighted downsampled image, same channel count as X
    """
    if X.ndim == 2:
        numerator = X * confidence
    else:
        numerator = X * confidence[:, :, np.newaxis]

    numerator = self_resize(numerator, target_w, target_h)
    denom = self_resize(confidence.copy(), target_w, target_h)

    # Avoid division by zero
    denom = np.maximum(denom, 1e-10)

    if numerator.ndim == 2:
        return numerator / denom
    else:
        return numerator / denom[:, :, np.newaxis]


def weighted_downsample_6ch(channels, confidence, target_w, target_h):
    """Confidence-weighted downsampling for 6 single-channel matrices.

    Merges channels into 2 groups of 3, applies weighted downsample,
    then splits back.

    Args:
        channels: list of 6 float32 HxW matrices
        confidence: float32 HxW confidence map
        target_w, target_h: target dimensions

    Returns:
        list of 6 downsampled HxW matrices
    """
    # Merge into two 3-channel images
    m1 = np.stack(channels[:3], axis=-1) * confidence[:, :, np.newaxis]
    m2 = np.stack(channels[3:], axis=-1) * confidence[:, :, np.newaxis]

    m1_re = self_resize(m1, target_w, target_h)
    m2_re = self_resize(m2, target_w, target_h)

    denom = self_resize(confidence.copy(), target_w, target_h)
    denom = np.maximum(denom, 1e-10)

    m1_re /= denom[:, :, np.newaxis]
    m2_re /= denom[:, :, np.newaxis]

    result = []
    for i in range(3):
        result.append(m1_re[:, :, i])
    for i in range(3):
        result.append(m2_re[:, :, i])
    return result


def outer_product_images(X, Y):
    """Upper triangle of per-pixel outer product of 3-channel images.

    Returns 6 matrices: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
    """
    result = []
    for i in range(3):
        for j in range(3):
            if i <= j:
                result.append(X[:, :, i] * Y[:, :, j])
    return result


def solve_ldl3(covar, residual):
    """Per-pixel LDL decomposition for 3x3 linear system (paper formula 7).

    Args:
        covar: list of 6 float32 HxW matrices (upper triangle of 3x3 covariance)
        residual: float32 HxWx3

    Returns:
        float32 HxWx3 solution
    """
    A11, A12, A13, A22, A23, A33 = covar
    b0 = residual[:, :, 0]
    b1 = residual[:, :, 1]
    b2 = residual[:, :, 2]

    d1 = A11
    L12 = A12 / d1
    d2 = A22 - L12 * A12
    L13 = A13 / d1
    L23 = (A23 - L13 * A12) / d2
    d3 = A33 - L13 * A13 - L23 * L23 * d2

    y1 = b0
    y2 = b1 - L12 * y1
    y3 = b2 - L13 * y1 - L23 * y2

    x3 = y3 / d3
    x2 = y2 / d2 - L23 * x3
    x1 = y1 / d1 - L12 * x2 - L13 * x3

    return np.stack([x1, x2, x3], axis=-1)


def smooth_upsample(X, target_w, target_h):
    """Progressive upsampling to target size."""
    src_h, src_w = X.shape[:2]
    log4ratio = 0.5 * max(np.log2(target_w / src_w), np.log2(target_h / src_h))
    num_steps = max(1, round(log4ratio))

    ratio_w = target_w / src_w
    ratio_h = target_h / src_h
    step_w = src_w * ratio_w / num_steps
    step_h = src_h * ratio_h / num_steps

    result = X
    for i in range(1, num_steps + 1):
        w = round(step_w * i)
        h = round(step_h * i)
        result = self_resize(result, w, h)
    return result


def refine_mask(mask, reference, kernel=256):
    """Full mask refinement pipeline (port of mask_refine.cpp main).

    Args:
        mask: float32 HxW probability map in [0, 1]
        reference: float32 HxWx3 RGB image in [0, 1]
        kernel: downsampling factor for small representation

    Returns:
        float32 HxW refined mask in [0, 1]
    """
    H, W = mask.shape

    # Step 1: Confidence map
    confidence = probability_to_confidence(mask)

    # Step 2: Weighted downsample reference and mask
    small_w = round(W / kernel)
    small_h = round(H / kernel)
    reference_small = weighted_downsample(reference, confidence.copy(), small_w, small_h)
    source_small = weighted_downsample(mask, confidence.copy(), small_w, small_h)

    # Step 3: Outer product and weighted downsample
    outer_ref = outer_product_images(reference, reference)
    outer_ref_small = weighted_downsample_6ch(outer_ref, confidence.copy(), small_w, small_h)

    tri_small = outer_product_images(reference_small, reference_small)

    # Covariance = E[xx^T] - mean(x)mean(x)^T
    covar = []
    for i in range(6):
        covar.append(outer_ref_small[i] - tri_small[i])

    # Step 4: Weighted cross-correlation
    ref_src = reference * mask[:, :, np.newaxis]
    var = weighted_downsample(ref_src, confidence.copy(), small_w, small_h)

    residual_small = var - source_small[:, :, np.newaxis] * reference_small

    # Add regularization to diagonal
    eps = 0.01
    for i in [0, 3, 5]:
        covar[i] = covar[i] + eps * eps

    # Step 5: Solve for affine coefficients
    affine = solve_ldl3(covar, residual_small)

    # Step 6: Compute residual
    residual = source_small - np.sum(affine * reference_small, axis=-1)

    # Step 7: Upsample affine and residual to full resolution
    affine_full = smooth_upsample(affine, W, H)
    residual_full = smooth_upsample(residual, W, H)

    # Step 8: Apply affine correction
    output = np.sum(affine_full * reference, axis=-1) + residual_full
    output = np.clip(output, 0.0, 1.0)

    return output


def refine_mask_with_bilateral(mask, reference, kernel=256, d=0, sigma_color=20, sigma_space=10):
    """Refine mask and apply bilateral filter.

    Args:
        mask: float32 HxW in [0, 1]
        reference: float32 HxWx3 RGB in [0, 1]

    Returns:
        float32 HxW refined mask in [0, 1]
    """
    output = refine_mask(mask, reference, kernel)
    output_255 = (output * 255).astype(np.float32)
    filtered = cv.bilateralFilter(output_255, d, sigma_color, sigma_space)
    return filtered / 255.0
