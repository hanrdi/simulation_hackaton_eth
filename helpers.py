import firedrake as fd
from pyadjoint import no_annotations


def ramp_interpolation(
    rho: fd.Function,
    ramp_p: fd.Constant = fd.Constant(30.0),
    val_1: fd.Constant = fd.Constant(1.0),
    val_0: fd.Constant = fd.Constant(0.0),
):
    """RAMP (Rational Approximation of Material Properties) penalization

    Args:
        rho (float): Volume fraction function
        ramp_p (Union[float, fd.Constant], optional): Penalization parameter. Defaults to 30.0. Pick value from > 0.0
        val_1 (float, optional): Function value for rho=1. Defaults to 1.0.
        val_0 (float, optional): Function value for rho=0. Defaults to 0.0.

    Returns:
        ufl.Expr: Penalized material property
    """
    assert ramp_p.dat.data[0] > 0, "ramp_p has to be positive"
    return (rho) / (fd.Constant(1.0) + ramp_p * (fd.Constant(1.0) - rho)) * (
        val_1 - val_0
    ) + val_0


def kappa_interpolation(
    rho: fd.Function, ramp_p: fd.Constant, *, k_fluid: float, k_solid: float
):
    """Generate the interpolation for the heat diffusion coefficient depending on the rho value.

    Args:
        rho (Union[fd.Function, fd.Constant]): firedrake function, phase of density liquid, 0<=rho<=1
        ramp_p (Union[float, fd.Constant]): penalisation parameter
        k_fluid (float): heat conductivity in the fluid
        k_solid (float): heat conductivity in the solid

    Returns:
        fd.Function: the interpolated value coefficient
    """
    return ramp_interpolation(rho, ramp_p, val_0=k_solid, val_1=k_fluid)


@no_annotations
def mesh_size(mesh: fd.Mesh, mode="max"):
    """Calculate the min and max mesh element sizes

    Args:
        mesh (fd.mesh): firedrake mesh
        mode (str, optional): min or max. Defaults to "max".

    Returns:
        float: the min or max size of the mesh
    """
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    h_sizes = fd.assemble(
        fd.CellDiameter(mesh) / fd.CellVolume(mesh) * fd.TestFunction(DG0) * fd.dx
    )
    with h_sizes.dat.vec as h_vec:
        if mode == "max":
            size = h_vec.max()[1]
        elif mode == "min":
            size = h_vec.min()[1]
        else:
            raise ValueError(f"Mode {mode} not supported")
    return size


def wall_drag_term(L: fd.Constant, Ht: fd.Constant):
    """Wall drag term for the non-dim problem

    Args:
        Ht (fd.Constant): Channel half thickness

    Returns:
        ufl.Operator: Wall drag term
    """
    return fd.Constant(5.0 * L**2) / (fd.Constant(2.0) * Ht**2)


def alpha(rhof, Ht, ramp_p, Htfactor, L):
    """
    Wall drag term with the Brinkmann term

    Args:
        rhof (fd.Function): Density/Volume fraction
        Ht (float): Channel half thickness
        ramp_p (float): Penalization parameter
        Htfactor (float): Division factor for the channel half thickness

    Returns:
        ufl.Operator: Wall drag term
    """
    val_1 = fd.Constant(0.0)
    val_0 = wall_drag_term(L, Ht / Htfactor)
    return (fd.Constant(1.0) - rhof) / (fd.Constant(1.0) + ramp_p * rhof) * (
        val_0 - val_1
    ) + val_1
