from dolfinx.fem import (
    Constant, Function, Expression,
    functionspace,
    locate_dofs_topological, dirichletbc,
    form,
)
from dolfinx.mesh import (
    exterior_facet_indices,
    uniform_refine, refine,
)
from ufl import (
    TrialFunction, TestFunction,
    inner, dot, cross,
    grad, curl, div,
    jump, dx, dS,
    conditional, gt,
    FacetNormal, CellDiameter,
)
from dolfinx.fem.petsc import (
    LinearProblem, NonlinearProblem,
    assemble_vector,
)

from dolfinx.io import VTXWriter
from petsc4py import PETSc
import numpy as np
from utils import netgen_refine, calc_hcurl_error


class AdaptiveMaxwell:
    def __init__(self, degree, problem, max_dofs):
        self.degree = degree
        self.problem = problem
        self.max_dofs = max_dofs
        self.linear_solver_parameters = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            }
        self.nonlinear_solver_parameters = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "ksp_type": "cg",
            "ksp_atol": 1e-9,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "snes_view": None,
            "snes_monitor": None,
            "ksp_monitor": None,
            }


    def solve(self, domain):
        f, g, alpha, beta, gamma, _ = self.problem(domain)
        domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
        boundary_facets = exterior_facet_indices(domain.topology)
        S = functionspace(domain, ("CG", self.degree))
        p, q = TrialFunction(S), TestFunction(S)
        apq = inner(beta*grad(p), grad(q))*dx
        lq = inner(-g, q)*dx
        boundary_dofs_S = locate_dofs_topological(S, domain.topology.dim-1, boundary_facets)
        linear_problem = LinearProblem(
            a = apq,
            L = lq,
            bcs = [dirichletbc(Constant(domain, 0.0), boundary_dofs_S, S)],
            petsc_options_prefix="linearProblem",
            petsc_options=self.linear_solver_parameters)
        ph = linear_problem.solve()
        V = functionspace(domain, (f"N{self.degree}curl", 1))
        uh, v = Function(V), TestFunction(V)
        F = inner(alpha(uh)*curl(uh), curl(v))*dx
        F += gamma*inner(beta*uh, v)*dx
        F -= inner(f, v)*dx
        F -= gamma*inner(beta*grad(ph), v)*dx
        boundary_dofs_V = locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
        u_bc = Function(V); u_bc.x.array[:] = 0.0
        nonlinear_problem = NonlinearProblem(
            F,
            uh,
            bcs = [dirichletbc(u_bc, boundary_dofs_V)],
            petsc_options_prefix="nonlinearProblem",
            petsc_options=self.nonlinear_solver_parameters)
        uh = nonlinear_problem.solve()
        return uh, ph
    

    def solve_uniform(self, domain, num_refine_uniform=3, geoModel=False, uref=False):
        dofs = []
        hcurl_errors = []
        for i in range(num_refine_uniform + 1):
            PETSc.Sys.Print(f'Uniform refinement level {i}/{num_refine_uniform}: ')
            uh, _ = self.solve(domain)
            dofs.append(uh.function_space.dofmap.index_map.size_global)
            if uref:
                hcurl_error = calc_hcurl_error(uref, uh, True)
            else:
                uex = self.problem(domain)[-1]
                hcurl_error = calc_hcurl_error(uex, uh)
            hcurl_errors.append(hcurl_error)
            PETSc.Sys.Print(f"Level: {i} | Dofs: {dofs[-1]} | Hcurl error: {hcurl_error:.8f}")
            if dofs[-1] > self.max_dofs:
                PETSc.Sys.Print(f'Uniform refinement stopped at level {i}, with Dofs {dofs[-1]}.')
                break
            CG1 = functionspace(domain, ('CG', 1, (3,)))
            ucg = Function(CG1)
            ucg.interpolate(uh)
            with VTXWriter(
                domain.comm, f"{self.problem.__name__}/n{self.degree}/uniform/{i}.bp", [ucg], engine="BP4"
            ) as vtx:
                vtx.write(0.0)
            if i < num_refine_uniform:
                if geoModel:
                    domain = netgen_refine(geoModel, domain)
                else:
                    domain = uniform_refine(domain)
        return uh, dofs, hcurl_errors


    def estimate(self, uh, ph):
        domain = uh.function_space.mesh
        f, g, alpha, beta, gamma, _ = self.problem(domain)
        DG0 = functionspace(domain, ('DG', 0))
        w = TestFunction(DG0)
        n = FacetNormal(domain)
        h = CellDiameter(domain)
        eta1_sq = Function(DG0)
        G1 = (
            inner(h**2 * (div(beta*grad(ph)) - g)**2, w)*dx
            + inner(h('+') * jump(beta*grad(ph), n)**2, w('+'))*dS
            + inner(h('-') * jump(beta*grad(ph), n)**2, w('-'))*dS
        )
        eta2_sq = Function(DG0)
        G2 = (
            inner(h**2 * div(beta * (grad(ph) - uh))**2, w)*dx
            + inner(h('+') * jump(beta * (grad(ph) - uh))**2, w('+'))*dS
            + inner(h('-') * jump(beta * (grad(ph) - uh))**2, w('-'))*dS
        )
        eta3_sq = Function(DG0)
        RT3 = f - curl(alpha(uh) * curl(uh)) + gamma * beta * (grad(ph) - uh)
        JF3 = jump(cross(alpha(uh) * curl(uh), n))
        G3 = (
            inner(h**2 * dot(RT3, RT3), w)*dx
            + inner(h('+') * dot(JF3, JF3), w('+'))*dS
            + inner(h('-') * dot(JF3, JF3), w('-'))*dS
        )
        assemble_vector(eta1_sq.x.petsc_vec, form(G1))
        assemble_vector(eta2_sq.x.petsc_vec, form(G2))
        assemble_vector(eta3_sq.x.petsc_vec, form(G3))
        eta1 = Function(DG0)
        eta1.x.array[:] = np.sqrt(eta1_sq.x.array[:])
        eta2 = Function(DG0)
        eta2.x.array[:] = np.sqrt(eta2_sq.x.array[:])
        eta3 = Function(DG0)
        eta3.x.array[:] = np.sqrt(eta3_sq.x.array[:])
        eta_sq = (eta1_sq.x.array + eta2_sq.x.array + eta3_sq.x.array)
        est_error = np.sqrt(eta_sq.dot(eta_sq))
        return eta1, eta2, eta3, est_error


    def mark(self, eta1, eta2, eta3, theta):
        DG0 = eta1.function_space
        eta1_Max = eta1.x.petsc_vec.max()[1]
        eta2_Max = eta2.x.petsc_vec.max()[1]
        eta3_Max = eta3.x.petsc_vec.max()[1]
        should_refine = conditional(gt(eta1, theta*eta1_Max), 1, 
                        conditional(gt(eta2, theta*eta2_Max), 1,
                        conditional(gt(eta3, theta*eta3_Max), 1, 0)))
        markers = Function(DG0)
        markers.interpolate(Expression(should_refine, DG0.element.interpolation_points))
        cells_to_mark = np.flatnonzero(np.isclose(markers.x.array.astype(np.int32), 1))
        return cells_to_mark


    def solve_adaptive(self, domain, num_refine_adaptive, geoModel=False, theta=0.6, ref=False):
        if ref:
            sols = []
        dofs = []
        hcurl_errors = []
        estimated_errors = []
        for i in range(num_refine_adaptive + 1):
            PETSc.Sys.Print(f'Adaptive refinement level {i}/{num_refine_adaptive}:')
            uh, ph = self.solve(domain)
            eta1, eta2, eta3, est_error = self.estimate(uh, ph)
            estimated_errors.append(est_error)
            dofs.append(uh.function_space.dofmap.index_map.size_global)
            if ref:
                sols.append(uh)
                PETSc.Sys.Print(f"Level: {i} | Dofs: {dofs[-1]} | Estimated error: {est_error:.8f}")
            else:
                uex = self.problem(domain)[-1]
                hcurl_error = calc_hcurl_error(uex, uh)
                hcurl_errors.append(hcurl_error)
                PETSc.Sys.Print(f"Level: {i} | Dofs: {dofs[-1]} | Hcurl error: {hcurl_error:.8f} | Estimated error: {est_error:.8f}")
            if dofs[-1] > self.max_dofs:
                PETSc.Sys.Print(f'Adaptive refinement stopped at level {i}, with Dofs {dofs[-1]}.')
                break
            CG1 = functionspace(domain, ('CG', 1, (3,)))
            ucg = Function(CG1)
            ucg.interpolate(uh)
            with VTXWriter(
                domain.comm, f"{self.problem.__name__}/n{self.degree}/adaptive/{i}.bp", [ucg], engine="BP4"
            ) as vtx:
                vtx.write(0.0)
            if i < num_refine_adaptive:
                cells_to_mark = self.mark(eta1, eta2, eta3, theta)
                if geoModel is not False:
                    domain = netgen_refine(geoModel, domain, cells_to_mark)
                else:
                    domain,_,__ = refine(domain, cells_to_mark)
        if ref:
            uref = sols.pop()
            dofs.pop()
            estimated_errors.pop()
            for i in range(len(sols)):
                hcurl_errors.append(calc_hcurl_error(uref, sols[i], True))
                PETSc.Sys.Print(f"Level: {i} | Dofs: {dofs[i]} | Hcurl error: {hcurl_errors[i]:.8f} | Estimated error: {estimated_errors[i]:.8f}")
        return uh, dofs, hcurl_errors, estimated_errors
