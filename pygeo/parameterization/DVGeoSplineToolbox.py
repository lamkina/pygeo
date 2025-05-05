from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np
from baseclasses.utils import Error
from mpi4py import MPI
from numpy.typing import NDArray
from splinetoolbox import curves, importGeometry, surfaces

from .. import pyGeo, pyNetwork
from .BaseDVGeo import BaseDVGeometry
from .designVars import geoDVGlobal, geoDVSectionLocal, geoDVShapeFunc

PathLike: TypeAlias = str | Path


class DVGeometrySplineToolbox(BaseDVGeometry):
    def __init__(self, filename: PathLike, name: str | None = None) -> None:
        super().__init__(fileName=filename, name=name)

    def addPointSet(
        self,
        points: NDArray[np.float64],
        ptName: str,
        origConfig: bool = True,
        coordXfer: Callable | None = None,
        activeChildren: list[str] | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop("compNames", None)  # compNames is only needed for DVGeometryMulti, so remove it if passed
