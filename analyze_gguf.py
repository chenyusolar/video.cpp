import struct


def align(pos, alignment=32):
    return (pos + alignment - 1) & ~(alignment - 1)


with open(r"E:\vllm-project\video.cpp\models\ltx-2.3-22b-dev-Q4_K_M.gguf", "rb") as f:
    header = f.read(24)
    tensor_count = struct.unpack("<Q", header[8:16])[0]
    kv_count = struct.unpack("<Q", header[16:24])[0]

    print(f"Tensors: {tensor_count}, KV: {kv_count}")

    # Read KV metadata
    for i in range(kv_count):
        key_len = struct.unpack("<Q", f.read(8))[0]
        f.read(key_len)
        val_type = struct.unpack("<I", f.read(4))[0]

        if val_type == 8:
            val_len = struct.unpack("<Q", f.read(8))[0]
            f.read(val_len)
        elif val_type == 4:
            f.read(4)
        elif val_type == 2:
            f.read(4)
        else:
            f.read(8)

    print(f"After KV, position = {f.tell()}")

    # Try different alignments to find correct one
    print("\nTensor 0 at pos 4927:")
    f.seek(4927)

    # Tensor 0 - using u64 for name_len
    name_len = struct.unpack("<Q", f.read(8))[0]
    name = f.read(name_len).decode("utf-8", errors="replace")
    n_dims = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)]
    dtype = struct.unpack("<I", f.read(4))[0]
    offset = struct.unpack("<Q", f.read(8))[0]
    print(f'  Using u64 for name_len: name_len={name_len}, name="{name}", dims={dims}')

    # After tensor 0 header
    pos_after = f.tell()
    print(f"  Position after tensor 0 header: {pos_after}, aligned: {align(pos_after)}")

    # Read tensor 1 at aligned position
    f.seek(align(pos_after))
    print(f"\nTensor 1 at pos {f.tell()}:")

    name_len_raw = f.read(8)
    name_len = struct.unpack("<Q", name_len_raw)[0]
    print(f"  Raw bytes: {name_len_raw.hex()}, as u64: {name_len}")

    # If name_len looks wrong, try u32
    if name_len > 1000 or name_len == 0:
        f.seek(align(pos_after))
        name_len = struct.unpack("<I", f.read(4))[0]
        print(f"  Using u32 for name_len: {name_len}")
        name = f.read(name_len).decode("utf-8", errors="replace")
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)]
        dtype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        print(f'  name="{name}", dims={dims}, dtype={dtype}')
    else:
        name = f.read(name_len).decode("utf-8", errors="replace")
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)]
        dtype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        print(f'  name="{name}", dims={dims}, dtype={dtype}')

    print(f"\nFinal position: {f.tell()}")
