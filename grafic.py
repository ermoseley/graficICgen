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

    def read(self, filename, dtype=np.float32):
        """
        Read a Grafic-format unformatted Fortran binary file.
        """
        with open(filename, "rb") as f:
            # Read header record
            header_len = struct.unpack("i", f.read(4))[0]
            header_values = struct.unpack("3i8f", f.read(header_len))
            f.read(4)  # trailing length

            self.header = {
                "n1": header_values[0],
                "n2": header_values[1],
                "n3": header_values[2],
                "dx": header_values[3],
                "xoff1": header_values[4],
                "xoff2": header_values[5],
                "xoff3": header_values[6],
                "f1": header_values[7],
                "f2": header_values[8],
                "f3": header_values[9],
                "f4": header_values[10],
            }

            # Read data record
            data_len = struct.unpack("i", f.read(4))[0]
            n1, n2, n3 = self.header["n1"], self.header["n2"], self.header["n3"]
            self.data = np.frombuffer(f.read(data_len), dtype=dtype).reshape((n1, n2, n3))
            f.read(4)  # trailing length
