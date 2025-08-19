
import sys
import numpy as np
import grafic

def write_uniform_field(filename, value, shape, size, dtype='f4', int_output=False):
    """Write a uniform 3D field to file."""
    field = grafic.Grafic()
    field.data = np.full(shape, value, dtype=dtype)
    field.make_header(size)
    if int_output:
        field.write_int64(filename)
    else:
        field.write_float(filename)


def write_array(filename, array, size, int_output=False):
    """Write a given array to file."""
    field = grafic.Grafic()
    field.data = array
    field.make_header(size)
    if int_output:
        field.write_int64(filename)
    else:
        field.write_float(filename)


if __name__ == '__main__':

    # Arguments
    lvl_max = int(sys.argv[1])     # refinement level
    size_cu = float(sys.argv[2])   # size of the box (in code units)
    size_ph = float(sys.argv[3])   # size of the box (in cm)
    Bz = float(sys.argv[4])        # magnetic field strength (code units)

    print(lvl_max, size_cu, size_ph, Bz)
    size = size_ph
    res1D = 2**lvl_max

    # Magnetic field
    Bx = 0.0
    By = 0.0

    # Write magnetic field components (left/right boundaries)
    # If you need non-uniform ICs, use write_array after generating the array
    # e.g.:
    # zeros = np.zeros((res1D, res1D, res1D), dtype='f4')
    # write_array('ic_u', zeros, size)
    write_uniform_field('ic_bxleft',  Bx, (res1D, res1D, res1D), size)
    write_uniform_field('ic_bxright', Bx, (res1D, res1D, res1D), size)
    write_uniform_field('ic_byleft',  By, (res1D, res1D, res1D), size)
    write_uniform_field('ic_byright', By, (res1D, res1D, res1D), size)
    write_uniform_field('ic_bzleft',  Bz, (res1D, res1D, res1D), size)
    write_uniform_field('ic_bzright', Bz, (res1D, res1D, res1D), size)

    # Write velocities. 
    write_uniform_field('ic_u', 0.0, (res1D, res1D, res1D), size)
    write_uniform_field('ic_v', 0.0, (res1D, res1D, res1D), size)
    write_uniform_field('ic_w', 0.0, (res1D, res1D, res1D), size)

    # Write densities & pressures
    write_uniform_field('ic_p', 1.0, (res1D, res1D, res1D), size)
    write_uniform_field('ic_d', 1.0, (res1D, res1D, res1D), size)

    # Particle IDs (integer, random permutation)
    ids = np.arange(1, res1D**3 + 1, dtype=np.int64)
    np.random.shuffle(ids)
    ids = ids.reshape(res1D, res1D, res1D)
    write_array('ic_particle_ids', ids, size, int_output=True)

    # Particle velocities
    write_uniform_field('ic_velcx', 0.0, (res1D, res1D, res1D), size)
    write_uniform_field('ic_velcy', 0.0, (res1D, res1D, res1D), size)
    write_uniform_field('ic_velcz', 0.0, (res1D, res1D, res1D), size)

    # Optional: particle masses
    # mconstant = dtg * np.log(10.0) * gfrac / (2.0 * nfam * (10.0**(0.5 * gfrac) - 1.0))
    # masses = mconstant * 10.0**(gfrac * ids / (nfam * res1D**3))
    # write_array('ic_massc', masses, size)


