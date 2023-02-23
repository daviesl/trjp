from rjlab.utils.barycenter import (
        bures_barycenter,
)
from rjlab.utils.kde import (
        kde_1D,
        kde_joint,
        plot_bivariates_scatter,
        plot_bivariates,
)
from rjlab.utils.linalgtools import (
        eprint,
        is_pos_def,
        is_pos_def_torch,
        make_pos_def_torch,
        make_pos_def,
        safe_cholesky,
        safe_logdet,
        safe_inv,
)
