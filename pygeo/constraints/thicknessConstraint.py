# External modules
import numpy as np

# Local modules
from .. import geo_utils
from .baseConstraint import GeometricConstraint


class ThicknessConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 2, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.D0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            for i in range(self.nCon):
                p1b, p2b = geo_utils.eDist_b(self.coords[2 * i, :], self.coords[2 * i + 1, :])
                if self.scaled:
                    p1b /= self.D0[i]
                    p2b /= self.D0[i]
                dTdPt[i, 2 * i, :] = p1b
                dTdPt[i, 2 * i + 1, :] = p2b

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ThicknessToChordConstraint(GeometricConstraint):
    """
    ThicknessToChordConstraint represents of a set of
    thickess-to-chord ratio constraints. One of these objects is
    created each time a addThicknessToChordConstraints2D or
    addThicknessToChordConstraints1D call is made. The user should not
    have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 4, lower, upper, scale, DVGeo, addToPyOpt)
        self.coords = coords

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.ToC0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            t = np.linalg.norm(self.coords[4 * i] - self.coords[4 * i + 1])
            c = np.linalg.norm(self.coords[4 * i + 2] - self.coords[4 * i + 3])
            self.ToC0[i] = t / c

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        ToC = np.zeros(self.nCon)
        for i in range(self.nCon):
            t = geo_utils.eDist(self.coords[4 * i], self.coords[4 * i + 1])
            c = geo_utils.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])
            ToC[i] = (t / c) / self.ToC0[i]

        funcs[self.name] = ToC

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dToCdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            for i in range(self.nCon):
                t = geo_utils.eDist(self.coords[4 * i], self.coords[4 * i + 1])
                c = geo_utils.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])

                p1b, p2b = geo_utils.eDist_b(self.coords[4 * i, :], self.coords[4 * i + 1, :])
                p3b, p4b = geo_utils.eDist_b(self.coords[4 * i + 2, :], self.coords[4 * i + 3, :])

                dToCdPt[i, 4 * i, :] = p1b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 1, :] = p2b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 2, :] = (-p3b * t / c**2) / self.ToC0[i]
                dToCdPt[i, 4 * i + 3, :] = (-p4b * t / c**2) / self.ToC0[i]

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dToCdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class KSMaxThicknessToChordConstraint(GeometricConstraint):
    def __init__(self, name, coords, rho, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        nCon = len(coords) // 4  # Divide the length of the coordinates by 4 to get the number of constraints
        super().__init__(name, nCon, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled
        self.rho = rho

        # Embed the coordinates
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Compute the t/c constraints
        self.ToC0 = np.zeros(self.nCon)

        for i in range(self.nCon):
            t = geo_utils.norm.eDist(self.coords[4 * i], self.coords[4 * i + 1])
            c = geo_utils.norm.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])

            # Divide by the chord that corresponds to this set of constraints
            self.ToC0[i] = t / c

        # Compute the absolute t/c at the baseline
        self.max0 = np.max(self.ToC0)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)

        # Compute the t/c constraints
        ToC = np.zeros(self.nCon)

        for i in range(self.nCon):
            # Calculate the thickness
            t = geo_utils.norm.eDist(self.coords[4 * i], self.coords[4 * i + 1])
            c = geo_utils.norm.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])

            # Divide by the chord that corresponds to this constraint section
            ToC[i] = t / c

            if self.scaled:
                ToC[i] /= self.ToC0[i]

        # Now we want to take the KS max over the toothpicks
        maxToC = geo_utils.KSfunction.compute(ToC, self.rho)
        funcs[self.name] = maxToC

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dToCdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            ToC = np.zeros(self.nCon)
            for i in range(self.nCon):
                t = geo_utils.eDist(self.coords[4 * i], self.coords[4 * i + 1])
                c = geo_utils.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])

                ToC[i] = t / c

                if self.scaled:
                    ToC[i] /= self.ToC0[i]

                p1b, p2b = geo_utils.eDist_b(self.coords[4 * i, :], self.coords[4 * i + 1, :])
                p3b, p4b = geo_utils.eDist_b(self.coords[4 * i + 2, :], self.coords[4 * i + 3, :])

                dToCdPt[i, 4 * i, :] = p1b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 1, :] = p2b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 2, :] = (-p3b * t / c**2) / self.ToC0[i]
                dToCdPt[i, 4 * i + 3, :] = (-p4b * t / c**2) / self.ToC0[i]

            # Get the derivative of the ks function with respect to the t/c constraints
            dKSdToC, _ = geo_utils.KSfunction.derivatives(ToC, self.rho)

            # Use the chain rule to compute the derivative
            # - Need a vector-tensor product
            # - dKSdToC is shape (1, nCon) and dToCdPt is shape (nCon, nCoords, 3)
            # - The shape of dKSdPt is (nCoords, 3)
            dKSdPt = np.einsum("ij,ijk->jk", dKSdToC.T, dToCdPt)

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dKSdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))
