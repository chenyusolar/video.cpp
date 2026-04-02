#!/usr/bin/env python3
import struct


def read_string_varint(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def align(pos, alignment=32):
    return (pos + alignment - 1) & ~(alignment - 1)


with open(r"E:\vllm-project\video.cpp\models\ltx-2.3-22b-dev-Q4_K_M.gguf", "rb") as f:
    # Read header
    magic = f.read(4)
    print(f"Magic: {magic}")

    version = struct.unpack("<I", f.read(4))[0]
    print(f"Version: {version}")

    tensor_count = struct.unpack("<Q", f.read(8))[0]
    print(f"Tensor count: {tensor_count}")

    metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
    print(f"Metadata KV count: {metadata_kv_count}")

    print(f"\nAfter header, position = {f.tell()}")

    # Read KV metadata
    for i in range(metadata_kv_count):
        key = read_string_varint(f)
        val_type = struct.unpack("<I", f.read(4))[0]

        if val_type == 8:  # string
            val = read_string_varint(f)
        elif val_type == 4:  # uint
            val = struct.unpack("<I", f.read(4))[0]
        elif val_type == 6:  # u64
            val = struct.unpack("<Q", f.read(8))[0]
        elif val_type == 7:  # i64
            val = struct.unpack("<q", f.read(8))[0]
        elif val_type == 11:  # array uint32
            arr_len = struct.unpack("<Q", f.read(8))[0]
            val = list(struct.unpack(f"<{arr_len}I", f.read(arr_len * 4)))
        else:
            val = f"type={val_type}"

        print(f"  KV {i}: {key} = {val}")

    tensor_data_offset = f.tell()
    print(f"\nTensor data offset (calculated): {tensor_data_offset}")

    # Now read first few tensors
    print("\n--- Reading first 5 tensors ---")
    for i in range(min(5, tensor_count)):
        name = read_string_varint(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)]
        offset = struct.unpack("<Q", f.read(8))[0]
        dtype = struct.unpack("<I", f.read(4))[0]

        dtype_names = {
            0: "F32",
            1: "F16",
            2: "BF16",
            3: "Q4_0",
            4: "Q4_1",
            5: "Q5_0",
            6: "Q5_1",
            7: "Q8_0",
            8: "Q2_K",
            9: "Q3_K",
            10: "Q4_K",
            11: "Q5_K",
            12: "Q6_K",
            13: "Q8_K",
        }
        dtype_name = dtype_names.get(dtype, f"UNKNOWN({dtype})")

        num_elements = 1
        for d in dims:
            num_elements *= d

        print(f"\nTensor {i}:")
        print(f"  name: {name}")
        print(f"  n_dims: {n_dims}, dims: {dims}")
        print(f"  num_elements: {num_elements}")
        print(f"  dtype: {dtype} ({dtype_name})")
        print(f"  offset (raw from file): {offset}")

        # Current position after reading header
        pos_after = f.tell()
        aligned_pos = align(pos_after)
        print(f"  position after header: {pos_after}, aligned: {aligned_pos}")

        # Seek to aligned position for next tensor
        f.seek(aligned_pos)

    # Go back to tensor_data_offset and verify data
    f.seek(tensor_data_offset)
    first_bytes = f.read(32)
    print(f"\n\nData at tensor_data_offset ({tensor_data_offset}):")
    print(f"  First 32 bytes: {first_bytes.hex()}")

    # Check a specific tensor at an offset
    if tensor_count > 1:
        # Read second tensor metadata properly
        f.seek(aligned_pos)
        name = read_string_varint(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<I", f.read(4))[0] for _ in range(n_dims)]
        offset = struct.unpack("<Q", f.read(8))[0]
        dtype = struct.unpack("<I", f.read(4))[0]

        print(f"\n\nTensor 2 (at aligned pos {aligned_pos}):")
        print(f"  name: {name}")
        print(f"  dims: {dims}")
        print(f"  offset: {offset}")
        print(f"  dtype: {dtype}")

        # Try to read data at this offset
        if offset > 0 and offset < 14326857120:
            f.seek(offset)
            data_bytes = f.read(min(32, 14326857120 - offset))
            print(f"  Data at offset {offset}: {data_bytes.hex()[:64]}...")
        else:
            print(f"  Offset {offset} is out of bounds (file size: 14326857120)")

    file_size = f.seek(0, 2)
    print(f"\n\nFile size: {file_size} bytes")
