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

class DataController:

  comm = rank = size = None

  data_arrays = data_attributes = None

  def __init__ ( self, inputpath, inputfile ):
    from mpi4py import MPI
    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

    if self.rank == 0:
      self.data_arrays = {}
      self.data_attributes = {}

      self.data_attributes['inputpath'] = inputpath
      self.data_attributes['inputfile'] = inputfile


  def data_dicts ( self ):
    return(self.data_arrays, self.data_attributes)


  def clean_data ( self ):
    print(self.data_arrays.keys())


  def read_external_files ( self ):
    # Read Input Data
    self.read_pao_inputfile()
    self.read_qe_output()

    # Broadcast Data
    self.broadcast_data_arrays()
    self.broadcast_data_attributes()

  def read_pao_inputfile ( self ):
    if self.rank == 0:
      from read_inputfile_xml_parse import read_inputfile_xml
      read_inputfile_xml(self.data_attributes['inputpath'], self.data_attributes['inputfile'], self)

  def read_qe_output ( self ):
    if self.rank == 0:
      from os.path import exists
      fpath = self.data_attributes['fpath']
      if exists(fpath+'/data-file.xml'):
        from read_QE_output_xml_parse import read_QE_output_xml
        read_QE_output_xml(self)
      elif exists(fpath+'/data-file-schema.xml'):
        from read_new_QE_output_xml_parse import read_new_QE_output_xml
        read_new_QE_output_xml(self)
      else:
        raise Exception('data-file.xml or data-file-schema.xml were not found.\n')


  def write_file_row_col ( self, fname, col1, col2 ):
    if self.rank == 0:
      from os.path import join
      if len(col1) != len(col2):
        print('Cannon write file: %s'%fname)
        print('Data does not have the same shape')
        self.comm.Abort()

      if self.data_attributes['verbose']:
        print('Writing file: %s'%fname)

      with open(join(self.data_attributes['inputpath'],fname), 'w') as f:
        for i in range(len(col1)):
          f.write('%.5f %.5e\n'%(col1[i],col2[i]))
    self.comm.Barrier()

  def write_transport_tensor ( self, fname, temp, energies, tensor ):
    if self.rank == 0:
      from os.path import join

      if self.data_attributes['verbose']:
        print('Writing transport file: %s'%fname)

      with open(join(self.data_attributes['inputpath'],fname), 'w') as f:
        for i in range(energies.size):
          tup = (temp,energies[i],tensor[0,0,i],tensor[1,1,i],tensor[2,2,i],tensor[0,1,i],tensor[0,2,i],tensor[1,2,i])
          f.write('%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e\n'%tup)
    self.comm.Barrier()

  def write_bxsf ( self, fname, bands, nbnd ):
    if self.rank == 0:
      from write2bxsf import write2bxsf

      if self.data_attributes['verbose']:
        print('Writing bxsf file: %s'%fname)

      write2bxsf(self, fname, bands, nbnd)
    self.comm.Barrier()


  def broadcast_single_array ( self, key, dtype=complex, root=0 ):
    import numpy as np
    ashape = self.comm.bcast((None if self.rank!=root else self.data_arrays[key].shape), root=root)
    if self.rank != root:
      self.data_arrays[key] = np.zeros(ashape, dtype=dtype, order='C')
    self.comm.Bcast(np.ascontiguousarray(self.data_arrays[key]), root=root)

  def broadcast_single_attribute ( self, key ):
    if self.rank == 0:
      for i in xrange(1,self.size):
        self.comm.send(self.data_attributes[key], dest=i)
    else:
      self.data_attributes[key] = self.comm.recv(source=0)

  def broadcast_data_attributes ( self ):
    if self.rank == 0:
      for i in xrange(1,self.size):
        self.comm.send(self.data_attributes, dest=i)
    else:
      self.data_attributes = self.comm.recv(source=0)

  def broadcast_data_arrays ( self ):
    if self.rank == 0:
      for i in xrange(1,self.size):
        self.comm.send(self.data_arrays, dest=i)
    else:
      self.data_arrays = self.comm.recv(source=0)

  def scatter_data_array ( self, key ):
    from communication import scatter_array
    self.data_arrays[key] = scatter_array(self.data_arrays[key])

  def gather_data_array ( self, key ):
    import numpy as np
    from communication import gather_array
    arr = self.data_arrays[key]
    aux = None
    size1 = arr.shape[0]

    if self.rank == 0:
      for i in xrange(1,self.size):
        size1 += comm.recv(source=i)
    else:
      comm.send(arr.shape[0], dest=0)

    if self.rank == 0:
      if len(arr.shape) > 1:
        aux = np.ndarray(shape=((size1,)+arr.shape[1:]), dtype=arr.dtype)
      else:
        aux = np.ndarray(shape=(size1), dtype=arr.dtype)

    gather_array(aux, arr)

    if self.rank == 0:
      self.data_arrays[key] = aux
    else:
      self.data_arrays[key] = None