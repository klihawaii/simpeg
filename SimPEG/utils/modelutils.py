from .matutils import mkvc, ndgrid, uniqueRows
import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.spatial import cKDTree, Delaunay
import scipy.sparse as sp


def surface2ind_topo(mesh, topo, gridLoc='CC', method='nearest', fill_value=np.nan):
    """
    Get active indices from topography

    Parameters
    ----------

    :param TensorMesh mesh: TensorMesh object on which to discretize the topography
    :param numpy.ndarray topo: [X,Y,Z] topographic data
    :param str gridLoc: 'CC' or 'N'. Default is 'CC'.
                        Discretize the topography
                        on cells-center 'CC' or nodes 'N'
    :param str method: 'nearest' or 'linear' or 'cubic'. Default is 'nearest'.
                       Interpolation method for the topographic data
    :param float fill_value: default is np.nan. Filling value for extrapolation

    Returns
    -------

    :param numpy.ndarray actind: index vector for the active cells on the mesh
                               below the topography
    """
    if mesh.dim == 1:
        raise NotImplementedError('surface2ind_topo not implemented' +
                                  ' for 1D mesh')
    if method == 'nearest':
        F = NearestNDInterpolator(topo[:, :-1], topo[:, -1])
        zTopo = F(mesh.gridCC[:, :-1])
    else:
        tri2D = Delaunay(topo[:, :-1])
        F = LinearNDInterpolator(tri2D, topo[:, -1])
        zTopo = F(mesh.gridCC[:, :-1])

        if any(np.isnan(zTopo)):
            F = NearestNDInterpolator(topo[:, :-1], topo[:, -1])
            zTopo[np.isnan(zTopo)] = F(mesh.gridCC[np.isnan(zTopo), :-1])

    if gridLoc == 'CC':

        # Fetch elevation at cell centers
        actind = mesh.gridCC[:, -1] < zTopo

    elif gridLoc == 'N':

        # Fetch elevation at cell centers
        actind = (mesh.gridCC[:, -1] + mesh.h_gridded[:, -1]/2.) < zTopo

    return mkvc(actind)


def surface_layer_index(mesh, topo, index=0):
    """
        Find the ith layer below topo
    """

    actv = np.zeros(mesh.nC, dtype='bool')
    # Get cdkTree to find top layer
    tree = cKDTree(mesh.gridCC)

    def ismember(a, b):
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
        return np.vstack([bind.get(itm, None) for itm in a])

    grid_x, grid_y = np.meshgrid(mesh.vectorCCx, mesh.vectorCCy)
    zInterp = mkvc(
        griddata(
            topo[:, :2], topo[:, 2], (grid_x, grid_y), method='nearest'
        )
    )

    # Get nearest cells
    r, inds = tree.query(np.c_[mkvc(grid_x), mkvc(grid_y), zInterp])
    inds = np.unique(inds)

    # Extract vertical neighbors from Gradz operator
    Dz = mesh._cellGradzStencil
    Iz, Jz, _ = sp.find(Dz)
    jz = np.sort(Jz[np.argsort(Iz)].reshape((int(Iz.shape[0]/2), 2)), axis=1)
    for ii in range(index):

        members = ismember(inds, jz[:, 1])
        inds = np.squeeze(jz[members, 0])

    actv[inds] = True

    return actv
