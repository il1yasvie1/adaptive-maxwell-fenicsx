from dolfinx.fem import (
    Function, functionspace,
    form, assemble_scalar, create_interpolation_data,
)

from ufl import (
    inner, curl, dx,
)
from mpi4py import MPI
import numpy as np


def calc_hcurl_error(u, uh, ref=False):
    if ref:
        domain_fine = u.function_space.mesh
        uh.x.scatter_forward()
        padding = 1e-16
        fine_mesh_cell_map = domain_fine.topology.index_map(domain_fine.topology.dim)
        num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
        cells = np.arange(num_cells_on_proc, dtype=np.int32)
        interpolation_data = create_interpolation_data(u.function_space, uh.function_space, cells, padding=padding)
        uh_interp = Function(u.function_space)
        uh_interp.interpolate_nonmatching(uh, cells, interpolation_data)
        uh = uh_interp
    error_form = form(inner(curl(u - uh), curl(u - uh))*dx + inner(u - uh, u - uh)*dx)
    error = np.sqrt(uh.function_space.mesh.comm.allreduce(assemble_scalar(error_form), MPI.SUM))
    return error


def netgen_refine(geoModel, domain, cells_to_mark=False):
    if cells_to_mark is not False:
        geoModel.ngmesh.Elements2D().NumPy()['refine'] = 0
    else:
        DG0 = functionspace(domain, ('DG', 0))
        markers = Function(DG0)
        markers.x.array[:] = 1.0
        cells_to_mark = np.flatnonzero(np.isclose(markers.x.array.astype(np.int32), 1))

    domain, (ct, ft) = geoModel.refineMarkedElements(
        domain.topology.dim,
        cells_to_mark,
        netgen_flags={'refine_faces': True})
    return domain
