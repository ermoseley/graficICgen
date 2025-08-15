import numpy as np
import struct
import os
import random


class Grafic:
    def __init__(self):
        self.data = None
        self.header = None
        self.ri = random.randint(10000, 99999)

    def make_header(self, box_size_cu):
        """Make header from data array and physical box size in code units."""
        if self.data is None:
            raise ValueError("No data array set for Grafic object")

        n1, n2, n3 = self.data.shape
        dx = float(box_size_cu) / n1
        self.header = [
            np.int32(n1),
            np.int32(n2),
            np.int32(n3),
            np.float32(dx),  # cell size in CU
            np.float32(0.0), # xoff1
            np.float32(0.0), # xoff2
            np.float32(0.0), # xoff3
            np.float32(0.0), # f1
            np.float32(0.0), # f2
            np.float32(0.0), # f3
            np.float32(0.0), # f4
        ]

    def _write_fortran_record(self, f, arr_bytes):
        """Write one unformatted Fortran sequential record."""
        f.write(struct.pack("i", len(arr_bytes)))
        f.write(arr_bytes)
        f.write(struct.pack("i", len(arr_bytes)))

    def _write_header(self, filename):
        """Write the Fortran-style header to temporary file."""
        head_bin = b"".join(
            struct.pack("i", v) if isinstance(v, np.int32)
            else struct.pack("f", v)
            for v in self.header
        )
        with open(filename, "wb") as f:
            self._write_fortran_record(f, head_bin)

    def _write_data(self, filename, dtype_out):
        """Write the data array in Fortran unformatted style."""
        # RAMSES expects the 3D cube to be written as n3 separate 2D slices
        data_t = self.data.transpose(2, 0, 1)  # shape (n3, n1, n2)
        for sli in data_t:
            sli_fortran = np.asfortranarray(sli.astype(dtype_out))
            self._write_fortran_record(open(filename, "ab"), sli_fortran.tobytes())

    def write(self, output_name):
        """Write data as float32 array."""
        tmpfile = f".grafic_tmp{self.ri:5d}"
        self._write_header(tmpfile)
        self._write_data(tmpfile, np.float32)
        os.rename(tmpfile, output_name)

    def write_int(self, output_name):
        """Write data as int64 array (e.g., for particle IDs)."""
        tmpfile = f".grafic_tmp{self.ri:5d}"
        self._write_header(tmpfile)
        self._write_data(tmpfile, np.int64)
        os.rename(tmpfile, output_name)

    def read(self, filename, dtype=None):
        """
        Read a GRAFIC file written with Fortran unformatted sequential I/O.

        Handles both layouts:
          (A) header + [n3 records of (n1 x n2)]  (what our writer does)
          (B) header + [1 record of (n1 x n2 x n3)]  (classic single-chunk)

        Parameters
        ----------
        filename : str
        dtype : np.dtype or None
            If None, infer from first data record size:
              4 bytes/elem -> float32 (or int32)
              8 bytes/elem -> int64
            You can override (e.g., dtype=np.float32) if needed.

        After return:
          self.header = [n1,n2,n3,dx,xoff1,xoff2,xoff3,f1,f2,f3,f4]
          self.data   = ndarray shape (n1,n2,n3)
        """
        def _read_record(fh):
            n = struct.unpack("i", fh.read(4))[0]
            payload = fh.read(n)
            n2 = struct.unpack("i", fh.read(4))[0]
            if n2 != n:
                raise IOError("Fortran record marker mismatch")
            return payload, n

        with open(filename, "rb") as f:
            # --- Header ---
            head_bytes, hlen = _read_record(f)
            # 3 * int32 + 8 * float32
            n1, n2, n3, dx, x1, x2, x3, f1, f2, f3, f4 = struct.unpack("3i8f", head_bytes)
            self.header = [n1, n2, n3, dx, x1, x2, x3, f1, f2, f3, f4]

            # --- First data record (peek) ---
            first_bytes, first_len = _read_record(f)

            # Infer dtype / layout if not provided
            if dtype is None:
                # Try per-slice record first
                if first_len == n1 * n2 * 4:
                    inferred = np.float32
                    layout = "sliced"
                    itemsize = 4
                elif first_len == n1 * n2 * 8:
                    inferred = np.int64
                    layout = "sliced"
                    itemsize = 8
                # Try single-chunk record
                elif first_len == n1 * n2 * n3 * 4:
                    inferred = np.float32
                    layout = "single"
                    itemsize = 4
                elif first_len == n1 * n2 * n3 * 8:
                    inferred = np.int64
                    layout = "single"
                    itemsize = 8
                else:
                    raise ValueError(
                        f"Cannot infer dtype/layout: record_len={first_len}, "
                        f"expected one of {{n1*n2*{4,8}, n1*n2*n3*{4,8}}}"
                    )
                dtype = inferred
            else:
                itemsize = np.dtype(dtype).itemsize
                if first_len == n1 * n2 * itemsize:
                    layout = "sliced"
                elif first_len == n1 * n2 * n3 * itemsize:
                    layout = "single"
                else:
                    raise ValueError(
                        f"Record length {first_len} not consistent with dtype {dtype} "
                        f"and grid ({n1},{n2},{n3})."
                    )

            if layout == "single":
                # Whole 3D array in one record, Fortran order on disk
                arr = np.frombuffer(first_bytes, dtype=dtype)
                self.data = arr.reshape((n1, n2, n3), order="F")
            else:
                # n3 slices, each (n1 x n2) in Fortran order
                slices = np.empty((n3, n1, n2), dtype=dtype)

                # first slice from the bytes we already read
                slices[0] = np.frombuffer(first_bytes, dtype=dtype).reshape((n1, n2), order="F")

                # remaining slices
                for k in range(1, n3):
                    rec_bytes, rec_len = _read_record(f)
                    if rec_len != first_len:
                        raise ValueError(f"Inconsistent slice record length at k={k}")
                    slices[k] = np.frombuffer(rec_bytes, dtype=dtype).reshape((n1, n2), order="F")

                # Reassemble to (n1, n2, n3)
                self.data = np.transpose(slices, (1, 2, 0))
