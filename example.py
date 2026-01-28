from dolfinx.fem import Constant, Function, functionspace
from ufl import (
    dot,
    curl,
    SpatialCoordinate,
    sin, cos, exp, pi,
    as_vector, conditional, gt, lt,
)
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from netgen.csg import OrthoBrick, Pnt, CSGeometry
import ngsPETSc.utils.fenicsx as ngfx
import argparse

from AdaptiveMaxwell import AdaptiveMaxwell
from utils import netgen_refine


def example1(domain):
    x, y, z = SpatialCoordinate(domain)
    uex = as_vector([pi*cos(pi*x)*sin(pi*y)*sin(pi*z),
                     pi*cos(pi*y)*sin(pi*z)*sin(pi*x),
                     pi*cos(pi*z)*sin(pi*y)*sin(pi*x)])
    alpha = lambda u: 1-.5/(1 + dot(curl(u), curl(u)))
    beta = 1.0
    f = Constant(domain, PETSc.ScalarType((0, 0, 0)))
    g = -3*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)
    N = domain.topology.index_map(domain.topology.dim).size_global
    gamma = Constant(domain, 1e9*N**(-1/3)) 
    return f, g, alpha, beta, gamma, uex


def example2(domain):
    x, y, z = SpatialCoordinate(domain)
    r = x**2 + y**2
    f = as_vector([0, 0, 100*conditional(lt(r, 1e-3), 1, 0)])
    g = Constant(domain, 0.0)
    def alpha(u):
        ncurlu = ncurlu = dot(curl(u), curl(u))
        return 1 - conditional(gt(x, 0), conditional(lt(x, 1), 1, 0), 0) / (4*(1+ncurlu)) - conditional(gt(y, 0), conditional(lt(y, 1), 1, 0), 0) / (4*(1+ncurlu))
    beta = 1.0
    N = domain.topology.index_map(domain.topology.dim).size_global
    gamma = Constant(domain, 1e4*N**(-1/3))
    return f, g, alpha, beta, gamma, False


def example3(domain):
    x, y, z = SpatialCoordinate(domain)
    uex = as_vector([0, 0, cos(.5*pi*x)*cos(.5*pi*y)*cos(.5*pi*z)])
    uexpr = lambda x : (np.zeros(x.shape[1]), np.zeros(x.shape[1]), np.cos(.5*pi*x[0])*np.cos(.5*pi*x[1])*np.cos(.5*pi*x[2]))
    def alpha(u, a0=1, a1=0.5, a2=1):
        ncurlu = dot(curl(u), curl(u))
        return a0 - a1*exp(-a2*ncurlu)
    beta = 1.0
    Vex = functionspace(domain, ('CG', 3, (3,)))
    u_interp = Function(Vex)
    u_interp.interpolate(uexpr)
    f = curl(alpha(u_interp)*curl(u_interp))
    g = -.5*pi*cos(.5*pi*x)*cos(.5*pi*y)*sin(.5*pi*z)
    N = domain.topology.index_map(domain.topology.dim).size_global
    gamma = Constant(domain, 10*N**(-1/3))
    return f, g, alpha, beta, gamma, uex


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--example', type=int, default=1)
    parser.add_argument('-d', '--degree', type=int, default=1)
    parser.add_argument('--domain', type=str, default='l-shape')
    parser.add_argument('-ni', '--num_refine_initial', type=int, default=2)
    parser.add_argument('-nu', '--num_refine_uniform', type=int, default=4)
    parser.add_argument('-na', '--num_refine_adaptive', type=int, default=4)
    parser.add_argument('--max_dofs', type=int, default=int(2e7))
    parser.add_argument('--theta', type=float, default=0.6)
    args = parser.parse_args()

    example_dict = dict({
        1: example1,
        2: example2,
        3: example3,
    })
    ref_dict = dict({
        1: False,
        2: True,
        3: False,
    })

    if args.domain == 'l-shape':
        cube1 = OrthoBrick(Pnt(-1,-1,0), Pnt(0,0,1))
        cube2 = OrthoBrick(Pnt(0,-1,0), Pnt(1,0,1))
        cube3 = OrthoBrick(Pnt(-1,0,0), Pnt(0,1,1))
        geo = CSGeometry()
        geo.Add(cube1 + cube2 + cube3)
        geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
        domain, (ct, ft), region_map = geoModel.model_to_mesh(gdim=3, hmax=2.0)
    elif args.domain == 'cube':
        cube = OrthoBrick(Pnt(-1,-1,-1), Pnt(1,1,1))
        geo = CSGeometry()
        geo.Add(cube)
        geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
        domain, (ct, ft), region_map = geoModel.model_to_mesh(gdim=3, hmax=2.0)
    else:
        raise NotImplementedError

    for i in range(args.num_refine_initial):
        domain = netgen_refine(geoModel, domain)

    eg = AdaptiveMaxwell(args.degree, example_dict[args.example], args.max_dofs)
    uh_adapt, dofs_adapt, hcurl_errors_adapt, estimated_errors_adapt = eg.solve_adaptive(domain, args.num_refine_adaptive, geoModel, args.theta, ref_dict[args.example])
    geoModel = ngfx.GeometricModel(geo, MPI.COMM_WORLD)
    domain, (ct, ft), region_map = geoModel.model_to_mesh(gdim=3, hmax=2.0)
    if ref_dict[args.example]:
        uh_unif, dofs_unif, hcurl_errors_unif = eg.solve_uniform(domain, args.num_refine_uniform, geoModel, uh_adapt)
    else:
        uh_unif, dofs_unif, hcurl_errors_unif = eg.solve_uniform(domain, args.num_refine_uniform, geoModel)

    with open(f'{example_dict[args.example].__name__}-n{args.degree}curl.txt', 'a') as f:
                f.write(f'{', '.join(str(_) for _ in dofs_adapt)}\n')
                f.write(f'{', '.join(str(_) for _ in hcurl_errors_adapt)}\n')
                f.write(f'{', '.join(str(_) for _ in estimated_errors_adapt)}\n')
                f.write(f'{', '.join(str(_) for _ in dofs_unif)}\n')
                f.write(f'{', '.join(str(_) for _ in hcurl_errors_unif)}\n')
