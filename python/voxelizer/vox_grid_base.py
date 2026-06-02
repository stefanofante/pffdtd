##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: vox_grid_base.py
#
# Description: Class for a voxel-grid for ray-tri / tri-box intersections
#  Uses multiprocessing
#
##############################################################################

import numpy as np
from numpy import array as npa
from common.timerdict import TimerDict
from common.tri_box_intersection import tri_box_intersection_vec
from common.room_geo import RoomGeo
from common.tris_precompute import tris_precompute
import multiprocessing as mp
from common.myfuncs import clear_dat_folder,clear_console
from common.myfuncs import get_default_nprocs
from tqdm import tqdm
import common.check_version as cv
import time

assert cv.ATLEASTVERSION38 #for shared memory (but project needs 3.9 anyway)

from multiprocessing import shared_memory

#base class for a voxel
class VoxBase:
    def __init__(self,bmin,bmax):
        self.bmin = bmin
        self.bmax = bmax
        self.tri_idxs = [] #triangle indices as list
        self.tris_pre = None
        self.tris_mat = None

#base class for a voxel grid
class VoxGridBase:
    def __init__(self,room_geo):
        tris = room_geo.tris
        pts = room_geo.pts
        tris_pre = room_geo.tris_pre
        mats = room_geo.mat_ind

        assert tris.ndim == 2
        assert pts.ndim == 2
        assert tris.shape[0] > tris.shape[1]
        assert pts.shape[0] > pts.shape[1]

        Npts = pts.shape[0]
        Ntris = tris.shape[0]

        self.tris = tris
        self.tris_pre = tris_pre
        self.mats = mats
        self.pts = pts
        self.Npts = Npts
        self.Ntris = Ntris

        self.voxels = []
        self.nonempty_idx = []
        self.timer = TimerDict()
        self.nprocs = get_default_nprocs()

    #fill the grid (primarily using tri-box intersections)
    def fill(self,Nprocs=None):
        if Nprocs is None:
            Nprocs = self.nprocs
        self.print(f'using {Nprocs} processes')

        tris = self.tris
        tris_pre = self.tris_pre
        Ntris = self.Ntris
        pts = self.pts
        Nvox = self.Nvox

        self.timer.tic('voxgrid fill')

        tri_pts = tris_pre['v']
        tri_bmin = tris_pre['bmin']
        tri_bmax = tris_pre['bmax']

        if Nvox==1:
            vox = self.voxels[0]
            vox.tri_idxs = np.arange(Ntris)
            vox.tris_pre = self.tris_pre
            vox.tris_mat = self.mats
            self.nonempty_idx = [0]
        else:
            if Nprocs>1:
                clear_dat_folder('mmap_dat')

            # Sprint 6.29: regular arrays + worker RETURN values instead of
            # shared_memory. The previous nested-closure + shared_memory pattern
            # breaks on Windows spawn (closures unpicklable; shared_memory
            # BufferError/PermissionError). Bit-exact: the per-voxel compute
            # (process_voxel body) is byte-for-byte unchanged.
            Ntris_vox = np.zeros(Nvox, dtype=np.int64)
            N_tribox_tests = np.zeros(Nvox, dtype=np.int64)

            #looping through boxes makes more sense because we append to voxels (for multithreading)
            def process_voxel(vox):
                candidates = np.nonzero(np.all(np.logical_and(vox.bmax >= tri_bmin,vox.bmin <= tri_bmax),axis=-1))[0]
                tri_idxs_vox = []
                hits = tri_box_intersection_vec(vox.bmin,vox.bmax,tris_pre[candidates])
                tri_idxs_vox = candidates[hits].tolist()
                return tri_idxs_vox, int(candidates.size)

            def process_voxels(vidx_list,proc_idx):
                # Returns {vox_idx: (ntris, ntribox_tests)}; writes the per-voxel
                # .dat files (cross-process safe). No shared-memory writes.
                out = {}
                pbar = tqdm(total=len(vidx_list),desc=f'process {proc_idx:02d} voxgrid processing',ascii=True,leave=False,position=0)
                for vox_idx in vidx_list:
                    tri_idxs_vox, n_tribox = process_voxel(self.voxels[vox_idx])
                    out[int(vox_idx)] = (len(tri_idxs_vox), n_tribox)
                    #if not empty, save vox data as file
                    if len(tri_idxs_vox)>0:
                        np.array(tri_idxs_vox,dtype=np.int64).tofile(f'mmap_dat/vox_{vox_idx}.dat')
                    pbar.update(1)

                pbar.close()
                return out

            
            if Nprocs==1: #keep separate for debug purposes
                #process without intermediate files
                pbar = tqdm(total=Nvox,desc=f'single process voxgrid processing',ascii=True,leave=False)
                for vox_idx in range(Nvox):
                    vox = self.voxels[vox_idx]
                    tri_idxs_vox, n_tribox = process_voxel(vox)
                    Ntris_vox[vox_idx] = len(tri_idxs_vox)
                    N_tribox_tests[vox_idx] = n_tribox
                    pbar.update(1)
                    if Ntris_vox[vox_idx]>0:
                        vox.tri_idxs = tri_idxs_vox
                        vox.tris_pre = self.tris_pre[vox.tri_idxs]
                        vox.tris_mat = self.mats[vox.tri_idxs]
                        assert Ntris_vox[vox_idx] == len(vox.tri_idxs)
                        self.nonempty_idx.append(vox_idx)
                pbar.close()

            elif Nprocs>1:
                vox_idx_lists = [[] for i in range(Nprocs)]
                vox_order = np.random.permutation(Nvox)
                #vox_order = np.arange(Nvox)
                for idx in range(Nvox):
                    cc = np.argmin([len(l) for l in vox_idx_lists])
                    vox_idx_lists[cc].append(vox_order[idx])

                # Sprint 6.29: dispatch via joblib loky (cloudpickle-serialises the
                # nested closure -> works on Windows spawn, unlike raw mp.Process).
                # Workers RETURN their per-voxel counts; we aggregate here. The
                # per-voxel .dat files (written by the workers) carry the tri lists.
                from joblib import Parallel, delayed
                partials = Parallel(n_jobs=int(Nprocs), backend="loky")(
                    delayed(process_voxels)(vox_idx_lists[proc_idx], proc_idx)
                    for proc_idx in range(Nprocs)
                )
                for part in partials:
                    for vidx, (ntris, ntribox) in part.items():
                        Ntris_vox[vidx] = ntris
                        N_tribox_tests[vidx] = ntribox

                #now load from temp files
                for vox_idx in range(Nvox):
                    vox = self.voxels[vox_idx]
                    if Ntris_vox[vox_idx]>0:
                        #now with one process read data from files
                        vox.tri_idxs = np.fromfile(f'mmap_dat/vox_{vox_idx}.dat',dtype=np.int64)
                        vox.tris_pre = self.tris_pre[vox.tri_idxs]
                        vox.tris_mat = self.mats[vox.tri_idxs]
                        assert Ntris_vox[vox_idx] == len(vox.tri_idxs)
                        self.nonempty_idx.append(vox_idx)

                clear_dat_folder('mmap_dat')

            self.print(self.timer.ftoc('voxgrid fill'))

            Ntris_vox_tot = np.sum(Ntris_vox)

            N_tribox_tests_tot = np.sum(N_tribox_tests)
            self.print(f'tribox checks={N_tribox_tests_tot} for {Ntris} tris and {Nvox} vox ({N_tribox_tests_tot/(Nvox*Ntris)*100.0:.2f} %)')

            # Sprint 6.29: shared_memory removed (regular arrays + worker returns).

            self.print(f'tris redundant={Ntris_vox_tot}, {100.*Ntris_vox_tot/self.Ntris:.2f} %')
            self.print(f'avg tris per voxel={Ntris_vox_tot/Nvox:.2f}')

    def print(self,fstring):
        print(f'--VOX_GRID_BASE: {fstring}')

    def print_stats(self):
        ntris_found = np.sum([len(vox.tri_idxs) for vox in self.voxels])
        self.print(f'total tris found in voxels={ntris_found:d}')

    #draws non-empty boxes only
    def draw_boxes(self,tube_radius,backend='mayavi'):
        from common.box import Box
        Nvox = self.Nvox
        self.print('drawing boxes..')
        boxtris = np.zeros((Nvox*12,3)) 
        boxpts = np.zeros((Nvox*8,3)) 
        tp = 0
        #build up a triangular mesh for all boxes in one go
        for i in range(len(self.nonempty_idx)):
            vox = self.voxels[self.nonempty_idx[i]]
            assert len(vox.tri_idxs)>0
            box = Box(*(vox.bmax-vox.bmin),shift=vox.bmin,centered=False)
            boxtris[i*12:(i+1)*12,:] = box.tris + tp
            boxpts[i*8:(i+1)*8,:] = box.verts 
            tp += 8
        self.print(f'{len(self.nonempty_idx)=}')
        self.print(f'{tp=}')

        if backend=='mayavi':
            from mayavi import mlab
            mlab.triangular_mesh(*(boxpts.T),boxtris,representation='mesh',color=(0,1,0),tube_radius=tube_radius)
            mlab.draw()
        elif backend=='polyscope':
            import polyscope as ps
            pmesh = ps.register_surface_mesh('voxels', boxpts, boxtris,color=(0,1,0),edge_color=(0,1,0),edge_width=tube_radius)
            #pmesh.set_transparency(0.0)

        self.print('boxes drawn..')
