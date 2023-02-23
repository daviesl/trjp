import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry.costs import Bures


def bures_barycenter(mu1, cov1, mu2, cov2, w1):
    # print (mu1.shape,mu2.shape,cov1.shape,cov2.shape)
    if len(np.array(mu1).flatten()) == 1:
        dim = 1
        mu1 = np.array(mu1).flatten()
        cov1 = np.abs(np.array(cov1).flatten())
        mu2 = np.array(mu2).flatten()
        cov2 = np.abs(np.array(cov2).flatten())
    else:
        assert mu1.shape[0] == mu2.shape[0] == cov1.shape[0] == cov2.shape[0]
        assert w1 >= 0 and w1 <= 1
        dim = mu1.shape[0]
    # print("BARYCENTER INPUT",mu1,cov1,mu2,cov2)
    if dim == 1:
        return (
            w1 * mu1 + (1 - w1) * mu2,
            (w1 * np.sqrt(cov1) + (1 - w1) * np.sqrt(cov2)) ** 2,
        )
    else:
        barycenter = Bures(dim)
        params = jnp.vstack(
            (jnp.concatenate((mu1, cov1.ravel())), jnp.concatenate((mu2, cov2.ravel())))
        )
        # print("dim",dim)
        # print("params",params)
        bc = barycenter.barycenter(jnp.array([w1, 1 - w1]), params)
        return np.array(bc[:dim]), np.array(bc[dim:].reshape((dim, dim)))


if __name__ == "__main__":
    import numpy as np

    mu1 = np.array([0.0, 0.0, 0.0])
    cov1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mu2 = np.array([1.0, 1.0, 1.0])
    cov2 = np.array([[2.0, 0.5, 0.5], [0.5, 2.0, 0.5], [0.5, 0.5, 2.0]])
    print(bures_barycenter(mu1, cov1, mu2, cov2, 0.5))

    # 1D test
    mu1 = np.array([0])
    cov1 = np.array([[100]])
    mu2 = np.array([50])
    cov2 = np.array([[1]])
    print("1D example\n")
    print(bures_barycenter(mu1, cov1, mu2, cov2, 0.5))
    # 2D test
    mu1 = np.array([0, 0])
    cov1 = np.array([[100, 50], [50, 100]])
    mu2 = np.array([50, 100])
    cov2 = np.array([[1, 0], [0, 1]])
    print("2D example\n")
    print(bures_barycenter(mu1, cov1, mu2, cov2, 0.5))
