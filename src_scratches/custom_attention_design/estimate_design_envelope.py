"""Scalar-only 4A.4.3 planning arithmetic; no ML imports, tensors or file I/O."""

from dataclasses import dataclass


@dataclass
class Count:
    parameters: int = 0
    macs: int = 0
    largest_conv_output: int = 0
    summed_conv_outputs: int = 0

    def conv(self, cin, cout, kernel, side, *, groups=1, norm=True, bias=False):
        weights = cin * cout * kernel * kernel // groups
        self.parameters += weights + (cout if bias else 0) + (2 * cout if norm else 0)
        self.macs += side * side * weights
        elements = side * side * cout
        self.largest_conv_output = max(self.largest_conv_output, elements)
        self.summed_conv_outputs += elements


def efficientnet_v2_s():
    # TorchVision v0.23.0 configs; counts include trainable BN affine terms,
    # not running-stat buffers. MACs omit normalization, pooling and activations.
    stages = (
        ("fused", 1, 1, 24, 24, 2),
        ("fused", 4, 2, 24, 48, 4),
        ("fused", 4, 2, 48, 64, 4),
        ("mb", 4, 2, 64, 128, 6),
        ("mb", 6, 1, 128, 160, 9),
        ("mb", 6, 2, 160, 256, 15),
    )
    count = Count()
    side = 112
    count.conv(3, 24, 3, side)
    taps = [(0, 24, side)]
    for stage_index, (kind, expansion, stride, cin, cout, repeats) in enumerate(stages, 1):
        for repeat in range(repeats):
            current_in = cin if repeat == 0 else cout
            current_stride = stride if repeat == 0 else 1
            expanded = current_in * expansion
            out_side = (side + current_stride - 1) // current_stride
            if kind == "fused":
                if expansion == 1:
                    count.conv(current_in, cout, 3, out_side)
                else:
                    count.conv(current_in, expanded, 3, out_side)
                    count.conv(expanded, cout, 1, out_side)
            else:
                if expanded != current_in:
                    count.conv(current_in, expanded, 1, side)
                count.conv(expanded, expanded, 3, out_side, groups=expanded)
                squeeze = max(1, current_in // 4)
                count.conv(expanded, squeeze, 1, 1, norm=False, bias=True)
                count.conv(squeeze, expanded, 1, 1, norm=False, bias=True)
                count.conv(expanded, cout, 1, out_side)
            side = out_side
        taps.append((stage_index, cout, side))
    count.conv(256, 1280, 1, side)
    taps.append((7, 1280, side))
    assert count.parameters + (1280 + 1) * 1000 == 21_458_488
    assert taps[3:] == [(3, 64, 28), (4, 128, 14), (5, 160, 14), (6, 256, 7), (7, 1280, 7)]
    return count, taps


def transformer_block_parameters(width):
    # Biased Q/K/V/output projections, two affine LayerNorms, ratio-4 FFN.
    return 12 * width * width + 13 * width


def residual_head(width, labels):
    parameters = (160 + 1280) * width + 9 * width**2 + 2 * width + labels * width + labels
    macs = 196 * 160 * width + 49 * 1280 * width + 196 * 9 * width**2 + 196 * labels * width
    return parameters, macs


def query_head(width, depth, labels, *, fine=False, mixer=False):
    channels = 64 if fine else 160
    local_tokens = 784 if fine else 196
    tokens = local_tokens + 49
    # Two projections/LNs, two scale embeddings, learned queries, final LN,
    # per-label output plus independent coarse global linear head.
    parameters = (channels + 1280) * width + 8 * width + 2 * labels * width + 1282 * labels
    parameters += depth * transformer_block_parameters(width)
    macs = local_tokens * channels * width + 49 * 1280 * width
    macs += depth * ((2 * tokens + 10 * labels) * width**2 + 2 * labels * tokens * width)
    macs += labels * width + 1280 * labels
    if mixer:
        parameters += transformer_block_parameters(width)
        macs += 12 * tokens * width**2 + 2 * tokens**2 * width
    return parameters, macs, tokens


def mib(elements, bytes_per_element=2):
    return elements * bytes_per_element / 2**20


def main():
    count, taps = efficientnet_v2_s()
    maxvit_trunk = 30_919_624 - (2 * 512 + 512 * 512 + 512 + 512 * 1000)
    assert maxvit_trunk == 30_143_944
    print("Scalar estimates only; no model construction or local memory measurements.")
    print(f"EfficientNetV2-S trunk: {count.parameters:,} parameters; {count.macs / 1e9:.6f} G convolution/SE MACs per image")
    print(f"EfficientNetV2-S + GAP/165: {count.parameters + 1281 * 165:,} parameters")
    print(f"MaxViT-T trunk: {maxvit_trunk:,}; + GAP/165: {maxvit_trunk + 513 * 165:,} parameters")
    print(f"Feature taps (index, channels, side): {taps}")
    print(f"Largest EfficientNet conv output: {count.largest_conv_output:,} elements/image; B4 fp16: {mib(4 * count.largest_conv_output):.3f} MiB")
    print(f"Sum of conv outputs: {count.summed_conv_outputs:,} elements/image; B4 fp16: {mib(4 * count.summed_conv_outputs):.3f} MiB (NOT a peak/upper bound)")
    for labels in (1, 50, 165):
        for width in (128, 256, 384):
            block_parts = 4 * (width**2 + width) + 4 * width + (8 * width**2 + 5 * width)
            assert transformer_block_parameters(width) == block_parts
            explicit_query = sum((160 * width, 1280 * width, 4 * width, 2 * width,
                                  labels * width, 2 * width, labels * width + labels,
                                  1280 * labels + labels, block_parts))
            assert query_head(width, 1, labels)[0] == explicit_query
    print("scale route L new_parameters total_parameters head_GMAC readout_score_MiB self_score_MiB")
    for name, width, heads, depth in (("S", 128, 4, 1), ("M", 256, 8, 2), ("L", 384, 12, 3)):
        assert width % heads == 0 and width // heads == 32 and width % 4 == 0
        for labels in (165, 50):
            params, macs = residual_head(width, labels)
            print(f"{name} residual {labels} {params} {count.parameters + params} {macs / 1e9:.6f} {mib(4 * labels * 196):.3f} -")
            for route, fine, mixer in (("query", False, False), ("query+mixer", False, True), ("query-fine", True, False)):
                params, macs, tokens = query_head(width, depth, labels, fine=fine, mixer=mixer)
                print(f"{name} {route} {labels} {params} {count.parameters + params} {macs / 1e9:.6f} {mib(4 * heads * labels * tokens):.3f} {mib(4 * heads * tokens * tokens) if mixer else 0:.3f}")
    print("Persistent FP32 state examples (not training peaks):")
    for name, params in (("query-S", count.parameters + query_head(128, 1, 165)[0]),
                         ("query-L+mixer", count.parameters + query_head(384, 3, 165, mixer=True)[0])):
        print(f"{name}: SGD+momentum {mib(params, 12):.2f} MiB; Adam-like {mib(params, 16):.2f} MiB")
    print(f"Early dense N=3136 scores, B4/h12/fp16: {mib(4 * 12 * 3136**2):.2f} MiB per layer")
    for width in (128, 256, 384):
        print(f"D={width}: B4 memory(245) {mib(4 * 245 * width):.3f} MiB; query FFN(165,4D) {mib(4 * 165 * 4 * width):.3f} MiB; optional spatial FFN(245,4D) {mib(4 * 245 * 4 * width):.3f} MiB")


if __name__ == "__main__":
    main()
