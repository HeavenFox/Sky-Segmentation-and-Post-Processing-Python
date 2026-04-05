"""Convert NCNN model weights to PyTorch state_dict.

Parses skysegsmall_sim-opt-fp16.param/.bin and saves a .pth file
that can be loaded directly into the U2NetP model.
"""

import struct
import numpy as np
import torch
from model import U2NetP


NCNN_FP16_MAGIC = 0x01306B47


def parse_param(param_path):
    """Parse NCNN .param file, return list of (name, num_output, weight_data_size, bias_term)
    for Convolution layers only, in file order."""
    conv_layers = []
    with open(param_path, 'r') as f:
        magic = f.readline().strip()
        counts = f.readline().strip()
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            layer_type = parts[0]
            layer_name = parts[1]
            if layer_type != 'Convolution':
                continue
            # Parse key=value params
            params = {}
            for p in parts[4:]:  # skip type, name, input_count, output_count
                if '=' in p:
                    k, v = p.split('=')
                    params[int(k)] = v
            num_output = int(params.get(0, 0))
            weight_data_size = int(params.get(6, 0))
            bias_term = int(params.get(5, 1))  # default 1
            conv_layers.append((layer_name, num_output, weight_data_size, bias_term))
    return conv_layers


def read_bin(bin_path, conv_layers):
    """Read NCNN .bin file and extract weights/biases for each Conv layer."""
    weights = []
    with open(bin_path, 'rb') as f:
        for name, num_output, weight_size, bias_term in conv_layers:
            # Read flag
            flag_bytes = f.read(4)
            flag = struct.unpack('<I', flag_bytes)[0]

            if flag == NCNN_FP16_MAGIC:
                # fp16 weights
                nbytes = weight_size * 2
                # Align to 4 bytes
                nbytes_aligned = (nbytes + 3) // 4 * 4
                raw = f.read(nbytes_aligned)
                w = np.frombuffer(raw[:nbytes], dtype=np.float16).astype(np.float32)
            elif flag == 0:
                # fp32 weights
                raw = f.read(weight_size * 4)
                w = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                # Flag might be first float of fp32 data
                remaining = f.read((weight_size - 1) * 4)
                w = np.frombuffer(flag_bytes + remaining, dtype=np.float32).copy()

            # Read bias
            if bias_term:
                bias_raw = f.read(num_output * 4)
                b = np.frombuffer(bias_raw, dtype=np.float32).copy()
            else:
                b = None

            weights.append((name, w, b))
    return weights


def build_state_dict(ncnn_weights, model):
    """Map NCNN weights to PyTorch state_dict keys by matching Conv layer order."""
    # Collect all Conv2d parameters in model, in module order
    conv_params = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_params.append((name, module))

    if len(conv_params) != len(ncnn_weights):
        raise ValueError(
            f"Conv layer count mismatch: model has {len(conv_params)}, "
            f"NCNN has {len(ncnn_weights)}"
        )

    state_dict = {}
    for (pt_name, pt_module), (ncnn_name, w, b) in zip(conv_params, ncnn_weights):
        out_ch = pt_module.out_channels
        in_ch = pt_module.in_channels
        kh = pt_module.kernel_size[0]
        kw = pt_module.kernel_size[1]
        expected_size = out_ch * in_ch * kh * kw

        if w.size != expected_size:
            raise ValueError(
                f"Weight size mismatch for {pt_name} (ncnn: {ncnn_name}): "
                f"expected {expected_size}, got {w.size} "
                f"(shape should be {out_ch}x{in_ch}x{kh}x{kw})"
            )

        w_tensor = torch.from_numpy(w.reshape(out_ch, in_ch, kh, kw))
        state_dict[pt_name + '.weight'] = w_tensor

        if b is not None:
            if b.size != out_ch:
                raise ValueError(
                    f"Bias size mismatch for {pt_name}: expected {out_ch}, got {b.size}"
                )
            state_dict[pt_name + '.bias'] = torch.from_numpy(b)

    return state_dict


def convert(param_path, bin_path, output_path):
    print(f"Parsing {param_path}...")
    conv_layers = parse_param(param_path)
    print(f"  Found {len(conv_layers)} Convolution layers")

    print(f"Reading {bin_path}...")
    ncnn_weights = read_bin(bin_path, conv_layers)

    print("Building PyTorch model...")
    model = U2NetP()
    conv_count = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Conv2d))
    print(f"  Model has {conv_count} Conv2d layers")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    print("Mapping weights...")
    state_dict = build_state_dict(ncnn_weights, model)

    # Verify by loading into model
    model.load_state_dict(state_dict, strict=False)
    # Check all keys matched
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    missing = model_keys - loaded_keys
    if missing:
        print(f"  WARNING: Missing keys: {missing}")
    extra = loaded_keys - model_keys
    if extra:
        print(f"  WARNING: Extra keys: {extra}")

    print(f"Saving to {output_path}...")
    torch.save(state_dict, output_path)
    print("Done!")


if __name__ == '__main__':
    convert(
        'skysegsmall_sim-opt-fp16.param',
        'skysegsmall_sim-opt-fp16.bin',
        'skyseg_u2netp.pth',
    )
