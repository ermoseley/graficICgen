import numpy as np
import struct

class Grafic:
    def __init__(self, endian="<"):
        if endian not in ("<", ">"):
            raise ValueError("endian must be '<' or '>'")
        self.endian = endian
        self.header = None
        self.data = None

    def _read_u32(self, f):
        b = f.read(4)
        if len(b) != 4:
            raise EOFError("Unexpected EOF while reading record marker")
        return int.from_bytes(b, "little" if self.endian == "<" else "big", signed=False)

    def read_header_only(self, path):
        with open(path, "rb") as f:
            reclen = self._read_u32(f)
            if reclen != 44:
                raise ValueError(f"Unexpected header length {reclen}, expected 44")
            header_bytes = f.read(reclen)
            if len(header_bytes) != reclen:
                raise EOFError("Truncated header")
            endlen = self._read_u32(f)
            if endlen != reclen:
                raise ValueError("Header record markers mismatch")

            n1, n2, n3, dx, x1, x2, x3, f1, f2, f3, f4 = struct.unpack(
                f"{self.endian}3i8f", header_bytes
            )
            self.header = [n1, n2, n3, dx, x1, x2, x3, f1, f2, f3, f4]
            return self.header

    def read(self, path, dtype=None):
        with open(path, "rb") as f:
            # Read header
            reclen = self._read_u32(f)
            if reclen != 44:
                raise ValueError(f"Unexpected header length {reclen}, expected 44")
            header_bytes = f.read(reclen)
            if len(header_bytes) != reclen:
                raise EOFError("Truncated header")
            endlen = self._read_u32(f)
            if endlen != reclen:
                raise ValueError("Header record markers mismatch")

            n1, n2, n3, dx, x1, x2, x3, f1, f2, f3, f4 = struct.unpack(
                f"{self.endian}3i8f", header_bytes
            )
            self.header = [n1, n2, n3, dx, x1, x2, x3, f1, f2, f3, f4]

            # Peek the first data record length to infer layout and itemsize
            pos = f.tell()
            try:
                first_data_len = self._read_u32(f)
            except EOFError:
                raise EOFError("No data section found after header")

            # Determine layout and itemsize
            total_elems_all = n1 * n2 * n3
            total_elems_slice = n1 * n2

            layout = None
            itemsize = None
            if first_data_len % total_elems_all == 0:
                # Could be 'single'
                itemsize = first_data_len // total_elems_all
                layout = "single"
            if first_data_len % total_elems_slice == 0:
                # Could be 'sliced' (take precedence if ambiguous with single and they are equal)
                slice_itemsize = first_data_len // total_elems_slice
                # Prefer 'sliced' if it makes sense (Grafic often writes sliced)
                layout = "sliced"
                itemsize = slice_itemsize

            if layout is None or itemsize not in (1, 2, 4, 8):
                raise ValueError(f"Cannot infer layout/itemsize from first record len {first_data_len}")

            # Infer dtype if not provided
            if dtype is None:
                if itemsize == 4:
                    dt = np.dtype(self.endian + "f4")  # default 4-byte => float32
                elif itemsize == 8:
                    dt = np.dtype(self.endian + "i8")  # default 8-byte => int64 (fix for your failing test)
                elif itemsize == 2:
                    # If you do not support 2-byte, you can raise instead
                    raise ValueError("2-byte elements unsupported without explicit dtype")
                elif itemsize == 1:
                    # Same note as above
                    raise ValueError("1-byte elements unsupported without explicit dtype")
            else:
                dt = np.dtype(dtype).newbyteorder(self.endian)
                if dt.itemsize != itemsize:
                    raise ValueError(
                        f"User dtype itemsize {dt.itemsize} does not match file element size {itemsize}"
                    )

            # Now actually read using detected layout and validated dtype
            f.seek(pos)  # rewind to first data record marker

            if layout == "single":
                payload_len = self._read_u32(f)
                expected = total_elems_all * dt.itemsize
                if payload_len != expected:
                    raise ValueError(f"Unexpected single-block payload length {payload_len}, expected {expected}")
                payload = f.read(payload_len)
                if len(payload) != payload_len:
                    raise EOFError("Truncated single-block data")
                endlen = self._read_u32(f)
                if endlen != payload_len:
                    raise ValueError("Single-block data record markers mismatch")

                arr = np.frombuffer(payload, dtype=dt, count=total_elems_all)
                arr = arr.reshape((n1, n2, n3), order="C")
                self.data = arr.copy()  # copy to detach from buffer
                return self.data

            else:  # sliced
                arr = np.empty((n1, n2, n3), dtype=dt)
                for k in range(n3):
                    slen = self._read_u32(f)
                    expected = total_elems_slice * dt.itemsize
                    if slen != expected:
                        raise ValueError(
                            f"Unexpected slice payload length {slen} for slice {k}, expected {expected}"
                        )
                    payload = f.read(slen)
                    if len(payload) != slen:
                        raise EOFError(f"Truncated data in slice {k}")
                    endlen = self._read_u32(f)
                    if endlen != slen:
                        raise ValueError(f"Slice {k} data record markers mismatch")

                    slice_arr = np.frombuffer(payload, dtype=dt, count=total_elems_slice)
                    arr[:, :, k] = slice_arr.reshape((n1, n2), order="C")
                self.data = arr
                return self.data
