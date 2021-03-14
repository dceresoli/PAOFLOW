# 
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

# contributed by Davide Ceresoli <dceresoli@gmail.com>
from scipy import fftpack as FFT
from scipy.io import FortranFile
import numpy as np


def read_QE_density(paoflow, spin=False, fft_grid=None):
    """Read the charge or spin density from a QE .dat file and transform to real space
    optionally on a larger FFT grid.
    
    usage: rho = read_QE_density(paoflow, spin=False, fft_grid=(32,32,32))

    """

    arry, attr = paoflow.data_controller.data_dicts()
    fname = attr['savedir']+'/charge-density.dat'
    if spin:
        fname = attr['savedir']+'/spin-density.dat'
        
    with FortranFile(fname, 'r') as f:
        gamma_only, ngm, nspin = f.read_ints(np.int32)
        bg = f.read_reals(np.float64).reshape(3,3,order='F')
        mill = f.read_ints(np.int32).reshape(3,ngm,order='F')
        rho = f.read_reals(np.complex128)

    gamma_only = (gamma_only != 0)
    omega = attr['omega']
    nr1, nr2, nr3 = arry['fft_rho']
    if fft_grid is not None:
        nr1, nr2, nr3 = fft_grid
    
    # transform to real space 
    rhog = np.zeros((nr1,nr2,nr3), dtype=complex)
    rhor = np.zeros((nr1,nr2,nr3), dtype=complex)

    for ig in range(ngm):
        rhog[mill[0,ig],mill[1,ig],mill[2,ig]] = rho[ig]
        if gamma_only:
            rhog[-mill[0,ig],-mill[1,ig],-mill[2,ig]] = np.conj(rho[ig])
            
    rhor = FFT.ifftn(rhog) * (nr1*nr2*nr3)
    return np.real(rhor)



def write_xsf(filename, paoflow, data=None):
    """Write the current crystal structure to file and optionally a 3d data set
    
    usage: write_xsf(filename, paoflow, data=None)
    
    """
    bohr = 0.52917721

    arry, attr = paoflow.data_controller.data_dicts()
    alat = attr['alat']
    a_vectors = arry['a_vectors']
    atoms = arry['atoms']
    tau = arry['tau']
    
    with open(filename, 'wt') as xsf:
        # write structure
        xsf.write('CRYSTAL\n')
        xsf.write('CONVVEC\n')
        for i in range(3):
            xsf.write(' %.14f %.14f %.14f\n' % tuple(a_vectors[i]*alat*bohr))

        xsf.write('CONVCOORD\n')
        xsf.write(str(len(atoms))+' 1\n')
        for na in range(len(atoms)):
            xsf.write(' %-2s' % atoms[na])
            xsf.write(' %20.14f %20.14f %20.14f' % tuple(tau[na]*bohr))
            xsf.write('\n')

        if data is None:
            return
 
        # if present write volumetric data
        xsf.write('BEGIN_BLOCK_DATAGRID_3D\n')
        xsf.write(' data\n')
        xsf.write(' BEGIN_DATAGRID_3Dgrid#1\n')

        data = np.asarray(data)
        if data.dtype == np.complex128:
            data = np.abs(data)**2

        shape = data.shape
        xsf.write('  %d %d %d\n' % (shape[0]+1, shape[1]+1, shape[2]+1))

        # origin and spanning vectors              
        xsf.write('  0.0 0.0 0.0\n')
        for i in range(3):
            xsf.write('  %.14f %.14f %.14f\n' % tuple(a_vectors[i]*alat*bohr))

        # 3d dataset
        for k in range(shape[2]+1):
            for j in range(shape[1]+1):
                xsf.write('   ')
                for i in range(shape[0]+1):
                    xsf.write('%12.8e ' % (data[i%shape[0], j%shape[1], k%shape[2]]))
                xsf.write('\n')

        xsf.write(' END_DATAGRID_3D\n')
        xsf.write('END_BLOCK_DATAGRID_3D\n')
 
    return

