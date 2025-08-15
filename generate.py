import sys
import numpy as np
import grafic
from my_units import *
from formulas import *


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
    density = float(sys.argv[3])   # density (in Msun)
    T = float(sys.argv[4])         # temperature (K)
    mu = float(sys.argv[5])        # mean molecular weight (MCs)
    Bz = float(sys.argv[6])        # magnetic field strength (code units)

    print(lvl_max, size_cu, density, T, mu, Bz)

    # Physical setup
    size = size_cu * scale_l       # cm
    mass = density * size**3       # grams
    c_s = sound_speed(T, mu, units_cgs)  # cm/s
    res1D = 2**lvl_max

    # Derived scales
    scale_t = scale_l / c_s
    scale_d = 7.0e-23
    scale_v = c_s
    scale_m = scale_d * scale_l**3

    # Magnetic field
    Bx = 0.0
    By = 0.0
    mu0 = 4.0 * np.pi  # in cgs
    v_alf = Bz / np.sqrt(mu0 * density)
    alf_mach = v_alf / c_s

    # Write magnetic field components (left/right boundaries)
    write_uniform_field('ic_bxleft',  Bx, (res1D, res1D, res1D), size)
    write_uniform_field('ic_bxright', Bx, (res1D, res1D, res1D), size)
    write_uniform_field('ic_byleft',  By, (res1D, res1D, res1D), size)
    write_uniform_field('ic_byright', By, (res1D, res1D, res1D), size)
    write_uniform_field('ic_bzleft',  Bz, (res1D, res1D, res1D), size)
    write_uniform_field('ic_bzright', Bz, (res1D, res1D, res1D), size)

    # Write velocities
    zeros = np.zeros((res1D, res1D, res1D), dtype='f4')
    write_array('ic_u', zeros, size)
    write_array('ic_v', zeros, size)
    write_array('ic_w', zeros, size)

    # Write densities & pressures
    ones = np.ones((res1D, res1D, res1D), dtype='f4')
    write_array('ic_d', ones, size)
    write_array('ic_p', ones, size)

    # Particle IDs (integer, random permutation)
    ids = np.arange(1, res1D**3 + 1, dtype=np.int64)
    np.random.shuffle(ids)
    ids = ids.reshape(res1D, res1D, res1D)
    write_array('ic_particle_ids', ids, size, int_output=True)

    # Particle velocities
    write_array('ic_velcx', zeros, size)
    write_array('ic_velcy', zeros, size)
    write_array('ic_velcz', zeros, size)

    # Optional: particle masses
    # mconstant = dtg * np.log(10.0) * gfrac / (2.0 * nfam * (10.0**(0.5 * gfrac) - 1.0))
    # masses = mconstant * 10.0**(gfrac * ids / (nfam * res1D**3))
    # write_array('ic_massc', masses, size)

    # Info file
    with open("../PARAMS_mhd.txt", 'w') as f:
        f.write(
            f"INPUT params\n"
            f"magnetic field strength Bx: {Bx} Gauss\n"
            f"magnetic field strength By: {By} Gauss\n"
            f"magnetic field strength Bz: {Bz} Gauss\n"
            f"alfven speed: {v_alf} cm/s\n"
            f"alfvenic mach number: {alf_mach}\n"
            f"sound speed: {c_s} cm/s\n"
        )
