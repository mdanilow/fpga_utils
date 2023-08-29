import argparse
import numpy as np


# x in NHWC shape
def to_binarray(x, bitwidth, folding=8, reverse_endian=True, reverse_inner=True, outpath='bytearray.bin'):

    # transpose to NCHW and fold channels
    x = x.astype('int32')
    x = x.transpose((0, 3, 1, 2))
    print(x[0, 1])
    sys.exit()
    N, C, H, W = x.shape
    channels_per_fold = C // folding
    x = x.reshape((N, folding, channels_per_fold, H, W))

    # change each number to u2 integer
    result = np.zeros(x.shape, dtype=int)
    mask = x < 0
    result[mask] = x[mask] + (1 << bitwidth)
    result[~mask] = x[~mask]

    # divide each number to bytes
    bytes_per_num = bitwidth // 8
    result_bytes = np.zeros(result.shape + (bytes_per_num,), dtype=int)
    print('result_bytes:', result_bytes.shape)
    for byte in range(bytes_per_num):
        result_bytes[..., byte] = (result >> byte*8) & 255

    # reverse bytes
    if reverse_endian:
        result_bytes = np.flip(result_bytes, axis=-1)
    # reverse channels in each fold
    if reverse_inner:
        result_bytes = np.flip(result_bytes, axis=2)

    # result_bytes in [batch, fold, channels_per_fold, h, w, bytes_per_num] shape
    # transpose to [batch, h, w, fold, channel_per_fold, bytes_per_num]
    result_bytes = result_bytes.transpose((0, 3, 4, 1, 2, 5))
    result_bytes = result_bytes.astype(np.uint8).tobytes()
    # result_bytes = bytearray(result_bytes)
    with open(outpath, "wb") as outfile:
        outfile.write(result_bytes)
    print(outpath, "saved")
    #     for n in range(N):
    #         for h in range(H):
    #             for w in range(W):
    #                 for fold in range(folding):
    #                     for channel in range(channels_per_fold):
    #                         for byte in range(bytes_per_num):
    #                             byte = result_bytes[n, fold, channel, h, w, byte]
    #                             # print(byte)
    #                             byte = bytes(byte)
    #                             outfile.write(byte)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a numpy array of NCHW shape into a binfile.')
    parser.add_argument('--inputfile', help='name(s) of input npy file(s) (i.e. "input.npy")', nargs="*", type=str, default=["example_input.npy"])
    args = parser.parse_args()

    for filename in args.inputfile:
        array = np.load(filename)
        to_binarray(array, bitwidth=8, folding=1)
