# Standard Python modules
import copy
from copy import deepcopy
import logging
from mimetypes import suffix_map
import os
from pathlib import Path
from typing import TypeAlias

# External modules
from baseclasses import tecplotIO as tpio
import numpy as np
from numpy import float64
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import factorized
from splinetoolbox import export, importGeometry, knotVectors, projections, utils
from splinetoolbox.surfaces import NURBSSurface

# Local modules
from . import geo_utils
from .topology import SurfaceTopology

# Type aliases
PathLike: TypeAlias = str | Path

# Setup logging
logger = logging.getLogger("pyGeo:pyGeo")


# ==============================================================================
# pyGeo Class
# ==============================================================================
class pyGeo:
    """
    pyGeo is a (fairly) complete geometry surfacing engine. It
    performs multiple functions including producing surfaces from
    cross sections and globally fitting surfaces.
    """

    def __init__(self) -> None:
        # ------------------- pyGeo Class Attributes -----------------
        self.topo: SurfaceTopology | None = None
        self.surfs: list[NURBSSurface] = []  # The list of BSpline or NURBS surfaces
        self.origSurfs: list[NURBSSurface] = []  # The list of original surfaces
        # objects
        self.nSurf = 0  # The total number of surfaces
        self.coef: npt.NDArray[np.float64] | None = None

    def addSplineSurfaces(self, surfs: list[NURBSSurface]) -> None:
        self.surfs.extend(surfs)
        self.origSurfs.extend(deepcopy(surfs))
        self.nSurf += len(surfs)

    def addPlot3DSurfaces(
        self,
        fileName: PathLike,
        order: np._OrderACF = "F",
        p: int = 3,
        q: int = 3,
        uKv: npt.NDArray[np.float64] | None = None,
        vKv: npt.NDArray[np.float64] | None = None,
    ) -> None:
        fileName = Path(fileName)
        assert fileName.suffix in [".xyz", ".x", ".p3d"], f"File {fileName} is not a Plot3D file"
        ctrlPntsList: list[npt.NDArray[np.float64]] = importGeometry.readPlot3DFaces(fileName, order)

        logger.info(f"Found {len(ctrlPntsList)} surfaces in the Plot3D file")
        surfs: list[NURBSSurface] = []
        for ctrlPnts in ctrlPntsList:
            if uKv is None:
                uKnotVec: npt.NDArray[np.float64] = knotVectors.computeUniformKnotVec(ctrlPnts.shape[0], p)
            else:
                uKnotVec = uKv

            if vKv is None:
                vKnotVec: npt.NDArray[np.float64] = knotVectors.computeUniformKnotVec(ctrlPnts.shape[1], q)
            else:
                vKnotVec = vKv

            if ctrlPnts.shape[-1] != 4:
                # Not in homogenous coordinates
                ctrlPnts = utils.combineCtrlPnts(ctrlPnts)

            surfs.append(NURBSSurface(p, q, uKnotVec, vKnotVec, ctrlPnts))

        logger.info(f"Adding {len(surfs)} NURBS surfaces from Plot3D file {fileName.name}")
        self.addSplineSurfaces(surfs)

    def addIGESSurfaces(self, fileName: PathLike) -> None:
        fileName = Path(fileName)
        assert fileName.suffix in [".iges", ".igs"], f"File {fileName} is not an IGES file"
        surfs: list[NURBSSurface] = importGeometry.readIGES(fileName).surfaces

        logger.info(f"Adding {len(surfs)} NURBS surfaces from IGES file {fileName}")
        self.addSplineSurfaces(surfs)

    # ----------------------------------------------------------------------------
    #               Initialization Type Functions
    # ----------------------------------------------------------------------------
    def fitGlobal(self):
        """
        Perform a global B-spline surface fit to determine the
        coefficients of each patch. This is only used with an plot3D
        init type
        """

        print("Global Fitting")
        nCtl = self.topo.nGlobal
        print(" -> Copying Topology")
        origTopo = copy.deepcopy(self.topo)

        print(" -> Creating global numbering")
        sizes = []
        for isurf in range(self.nSurf):
            sizes.append([self.surfs[isurf].Nu, self.surfs[isurf].Nv])

        # Get the Global number of the original data
        origTopo.calcGlobalNumbering(sizes)
        N = origTopo.nGlobal
        print(" -> Creating global point list")
        pts = np.zeros((N, 3))
        for ii in range(N):
            pts[ii] = self.surfs[origTopo.gIndex[ii][0][0]].X[origTopo.gIndex[ii][0][1], origTopo.gIndex[ii][0][2]]

        # Get the maximum k (ku, kv for each surf)
        kmax = 2
        for isurf in range(self.nSurf):
            if self.surfs[isurf].ku > kmax:
                kmax = self.surfs[isurf].ku
            if self.surfs[isurf].kv > kmax:
                kmax = self.surfs[isurf].kv

        nnz = N * kmax * kmax
        vals = np.zeros(nnz)
        rowPtr = [0]
        colInd = np.zeros(nnz, "intc")

        for ii in range(N):
            isurf = origTopo.gIndex[ii][0][0]
            i = origTopo.gIndex[ii][0][1]
            j = origTopo.gIndex[ii][0][2]

            u = self.surfs[isurf].U[i, j]
            v = self.surfs[isurf].V[i, j]

            vals, colInd = self.surfs[isurf].getBasisPt(u, v, vals, rowPtr[ii], colInd, self.topo.lIndex[isurf])

            kinc = self.surfs[isurf].ku * self.surfs[isurf].kv
            rowPtr.append(rowPtr[-1] + kinc)

        # Now we can crop out any additional values in col_ptr and vals
        vals = vals[: rowPtr[-1]]
        colInd = colInd[: rowPtr[-1]]
        # Now make a sparse matrix

        NN = sparse.csr_matrix((vals, colInd, rowPtr))
        print(" -> Multiplying N^T * N")
        NNT = NN.T
        NTN = NNT * NN
        print(" -> Factorizing...")
        solve = factorized(NTN)
        print(" -> Back Solving...")
        self.coef = np.zeros((nCtl, 3))
        for idim in range(3):
            self.coef[:, idim] = solve(NNT * pts[:, idim])

        print(" -> Setting Surface Coefficients...")
        self._updateSurfaceCoef()

    # ----------------------------------------------------------------------
    #                     Topology Information Functions
    # ----------------------------------------------------------------------

    def doConnectivity(self, fileName: PathLike | None = None, nodeTol: float = 1e-4, edgeTol: float = 1e-4) -> None:
        """
        This is the only public edge connectivity function.
        If fileName exists it loads the file OR it calculates the connectivity
        and saves to that file.

        Parameters
        ----------
        fileName : PathLike | None
            Filename for con file
        nodeTol : float
            The tolerance for identical nodes, by default 1e-4
        edgeTol : float
            The tolerance for midpoint of edges being identical, by default 1e-4
        """
        if fileName is not None and os.path.isfile(fileName):
            logger.info("Reading Connectivity File: %s" % (fileName))
            self.topo = SurfaceTopology(fileName=fileName)

            sizes = []
            for isurf in range(self.nSurf):
                sizes.append([self.surfs[isurf].nCtlu, self.surfs[isurf].nCtlv])
                self.surfs[isurf].computeData(recompute=True)
            self.topo.calcGlobalNumbering(sizes)
        else:
            self._calcConnectivity(nodeTol, edgeTol)
            assert self.topo is not None, "Topology is not properly initialized"
            sizes = []
            for isurf in range(self.nSurf):
                sizes.append([self.surfs[isurf].nCtlu, self.surfs[isurf].nCtlv])
            self.topo.calcGlobalNumbering(sizes)

            if fileName is not None:
                logger.info("Writing Connectivity File: %s" % (fileName))
                self.topo.writeConnectivity(fileName)

        self.setSurfaceCoef()

    def _calcConnectivity(self, nodeTol, edgeTol):
        """This function attempts to automatically determine the connectivity
        between the patches"""

        # Calculate the 4 corners and 4 midpoints for each surface

        coords: npt.NDArray[float64] = np.zeros((self.nSurf, 8, 3))

        for isurf in range(self.nSurf):
            beg, mid, end = self.origSurfs[isurf].getValueCorner(0, 0)
            coords[isurf][0] = beg
            coords[isurf][1] = end
            coords[isurf][4] = mid
            beg, mid, end = self.origSurfs[isurf].getValueCorner(1, 0)
            coords[isurf][2] = beg
            coords[isurf][3] = end
            coords[isurf][5] = mid
            beg, mid, end = self.origSurfs[isurf].getValueCorner(1, 1)
            coords[isurf][6] = mid
            beg, mid, end = self.origSurfs[isurf].getValueCorner(0, 1)
            coords[isurf][7] = mid

        self.topo = SurfaceTopology(coords=coords, nodeTol=nodeTol, edgeTol=edgeTol)

    def printConnectivity(self) -> None:
        """
        Print the Edge connectivity to the screen
        """
        assert self.topo is not None, "Topology is not properly initialized"
        self.topo.printConnectivity()

    # ==============================================================================
    # Surface Output Functions
    # ==============================================================================
    def writeTecplot(
        self,
        fileName: PathLike,
        orig: bool = False,
        surfs: bool = True,
        coef: bool = True,
        directions: bool = False,
    ) -> None:
        """Write the pyGeo Object to Tecplot file

        Parameters
        ----------
        fileName : PathLike
            File name for tecplot file. Should have .dat or .plt extension
        orig : bool, default=False
            Flag to write original surface data
        surfs : bool, default=True
            Flag to write discrete approximation of the actual surface
        coef : bool, default=True
            Flag to write b-spline coefficients
        directions : bool, default=False
            Flag to write surface direction visualization
        """

        assert self.topo is not None, "Topology is not properly initialized"
        fileName = Path(fileName)
        assert fileName.suffix in [".dat", ".plt"], f"File {fileName} is not a Tecplot file"

        # Write out the Interpolated Surfaces
        surfZones: list[tpio.TecplotZone] = []
        coefZones: list[tpio.TecplotZone] = []
        origZones: list[tpio.TecplotZone] = []
        directionZones: list[tpio.TecplotZone] = []

        if surfs:
            for isurf in range(self.nSurf):
                self.surfs[isurf].computeData()
                zone = tpio.TecplotOrderedZone(
                    name=f"Surf_{isurf}",
                    data={
                        "CoordinateX": self.surfs[isurf].data[..., 0],
                        "CoordinateY": self.surfs[isurf].data[..., 1],
                        "CoordinateZ": self.surfs[isurf].data[..., 2],
                    },
                    solutionTime=0,
                )
                surfZones.append(zone)

        # Write out the Control Points
        if coef:
            for isurf in range(self.nSurf):
                zone = tpio.TecplotOrderedZone(
                    name=f"ControlPointsW_{isurf}",
                    data={
                        "CoordinateX": self.surfs[isurf].ctrlPntsW[..., 0],
                        "CoordinateY": self.surfs[isurf].ctrlPntsW[..., 1],
                        "CoordinateZ": self.surfs[isurf].ctrlPntsW[..., 2],
                    },
                    solutionTime=0,
                )
                coefZones.append(zone)

        # Write out the Original Data
        if orig:
            for isurf in range(self.nSurf):
                self.origSurfs[isurf].computeData()
                zone = tpio.TecplotOrderedZone(
                    name=f"OriginalData_{isurf}",
                    data={
                        "CoordinateX": self.origSurfs[isurf].data[..., 0],
                        "CoordinateY": self.origSurfs[isurf].data[..., 1],
                        "CoordinateZ": self.origSurfs[isurf].data[..., 2],
                    },
                    solutionTime=0,
                )
                origZones.append(zone)

        # Write out The Surface Directions
        if directions:
            for isurf in range(self.nSurf):
                data = np.zeros((4, 3))
                data[0] = self.surfs[isurf].ctrlPntsW[1, 2]
                data[1] = self.surfs[isurf].ctrlPntsW[1, 1]
                data[2] = self.surfs[isurf].ctrlPntsW[2, 1]
                data[3] = self.surfs[isurf].ctrlPntsW[3, 1]
                zone = tpio.TecplotOrderedZone(
                    name=f"SurfaceDirections_{isurf}",
                    data={
                        "CoordinateX": data[..., 0],
                        "CoordinateY": data[..., 1],
                        "CoordianteZ": data[..., 2],
                    },
                    solutionTime=0,
                )
                directionZones.append(zone)

        zonesToWrite = surfZones + coefZones + origZones + directionZones
        tpio.writeTecplot(fileName, title="pyGeo", zones=zonesToWrite, precision="SINGLE")

    def writeIGES(self, fileName: PathLike, units: str = "m") -> None:
        """
        Write the surface to IGES format

        Parameters
        ----------
        fileName : PathLike
            File name of iges file. Should have .igs extension.
        units : str, default="m"
            Units for the iges file
        """
        fileName = Path(fileName)
        assert fileName.suffix in [".iges", ".igs"], f"File {fileName} is not an IGES file"
        export.writeIGES(fileName, self.surfs, units=units, productID="pyGeo", author="pyGeo")

    # ----------------------------------------------------------------------
    #                Update and Derivative Functions
    # ----------------------------------------------------------------------

    def _updateSurfaceCoef(self) -> None:
        """Copy the pyGeo list of control points back to the surfaces"""
        for ii in range(len(self.coef)):
            for jj in range(len(self.topo.gIndex[ii])):
                isurf = self.topo.gIndex[ii][jj][0]
                i = self.topo.gIndex[ii][jj][1]
                j = self.topo.gIndex[ii][jj][2]
                self.surfs[isurf].ctrlPntsW[i, j] = self.coef[ii].astype("d")

        for isurf in range(self.nSurf):
            self.surfs[isurf].setEdgeCurves()

    def setSurfaceCoef(self) -> None:
        """Set the surface coef list from the pyspline surfaces"""
        self.coef = np.zeros((self.topo.nGlobal, 4))
        for isurf in range(self.nSurf):
            surf = self.surfs[isurf]
            for i in range(surf.nCtlu):
                for j in range(surf.nCtlv):
                    self.coef[self.topo.lIndex[isurf][i, j]] = surf.ctrlPntsW[i, j]

    def getBounds(
        self, surfIndices: npt.NDArray[np.intc] | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Determine the extents of the collection of surfaces

        Parameters
        ----------
        surfIndices : npt.NDArray[np.intc] | None
            Indices of surfaces defining subset for which to get the bounding
            box, if None, all surfaces are used

        Returns
        -------
        xMin : array of length 3
            Lower corner of the bounding box
        xMax : array of length 3
            Upper corner of the bounding box
        """
        if surfIndices is None:
            surfIndices = np.arange(self.nSurf)

        # Get bounds for all surfaces at once
        bounds: list[tuple[npt.NDArray[np.float64], npt.NDArray[npt.float64]]] = [
            self.surfs[i].getBounds() for i in surfIndices
        ]
        mins, maxs = zip(*bounds)

        # Stack into arrays and find min/max along first axis
        Xmin0 = np.min(np.vstack(mins), axis=0)
        Xmax0 = np.max(np.vstack(maxs), axis=0)

        return Xmin0, Xmax0

    def projectCurve(
        self, curve: CurveType, surfaceIndices: Sequence[int] | None = None
    ) -> tuple[list[tuple[float, float, float, float]], list[int]]:
        """
        Project a curve object onto all or a subset of the surfaces.

        Parameters
        ----------
        curve : SplineToolbox.curves.CurveType
            Curve to use for the intersection calculation.
        surfaceIndices : Sequence[int] | None
            Indices of surfaces defining subset for which to use for
            projection, if None, all surfaces are used

        Returns
        -------
        result: list[tuple[float, float, float, float]]
            List of tuples containing the u and v parameters of the closest
            point on each surface, the s parameter of the closest point on
            the curve, and the distance to the closest point. Sorted by
            distance to the closest point.
        patchID: list[int]
            List of indices of the surface that the closest point is on.
            Sorted by distance to the closest point.

        Notes
        -----
        This algorithm performs a curve-surface projection for each surface.
        This can become intensive for a large number of surfaces.
        """

        if surfaceIndices is None:
            surfaceIndices = np.arange(self.nSurf)

        temp = []
        result = []
        patchID = []

        for i in range(len(surfaceIndices)):
            isurf = surfaceIndices[i]
            u, v, s, d = projections.curveSurface(curve, self.surfs[isurf])
            temp.append((u, v, s, np.linalg.norm(d)))

        # Sort the results by distance
        index = sorted(temp, key=lambda x: x[3])
        result = sorted(temp, key=lambda x: x[3])
        patchID = sorted(range(len(temp)), key=lambda x: temp[x][3])

        return result, patchID

    def projectPoints(self, points, *args, surfs=None, **kwargs):
        """Project on or more points onto the nearest surface.

        Parameters
        ----------
        points : list or array
            Singe point (size 3) or list of points size (N,3) points
            to project onto the surfaces

        surfs : list or array
            Indices of surface defining subset for which to use for
            projection

        Returns
        -------
        u : float or array
            u parameter values of closest point
        v : float or array
            v parameter values of closest point
        PID : int or int array
            Patch index corresponding to the u,v parameter values
        """

        if surfs is None:
            surfs = np.arange(self.nSurf)

        N = len(points)
        U = np.zeros((N, len(surfs)))
        V = np.zeros((N, len(surfs)))
        D = np.zeros((N, len(surfs), 3))
        for i in range(len(surfs)):
            isurf = surfs[i]
            U[:, i], V[:, i], D[:, i, :] = self.surfs[isurf].projectPoint(points, *args, **kwargs)

        u = np.zeros(N)
        v = np.zeros(N)
        patchID = np.zeros(N, "intc")

        # Now post-process to get the lowest one
        for i in range(N):
            d0 = np.linalg.norm(D[i, 0])
            u[i] = U[i, 0]
            v[i] = V[i, 0]
            patchID[i] = surfs[0]
            for j in range(len(surfs)):
                if np.linalg.norm(D[i, j]) < d0:
                    d0 = np.linalg.norm(D[i, j])
                    u[i] = U[i, j]
                    v[i] = V[i, j]
                    patchID[i] = surfs[j]

        return u, v, patchID
