import torch
from torch import nn
from torch import optim
import nflows as n
from nflows.utils import torchutils
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal, ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform, Transform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.base import InputOutsideDomain
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from torch.optim.lr_scheduler import *

from rjlab.distributions import *
from rjlab.utils.linalgtools import *

# from mixbeta import *
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import graphical_lasso
from scipy.special import factorial, logsumexp

# from rjlab.transforms.ptunif import *
import matplotlib.pyplot as plt
import numpy as np

PLOT_PROGRESS = False


def t2n(x):
    return x.detach().numpy().astype(np.float64)


def n2t(x):
    return torch.tensor(x, dtype=torch.float32)


class PassthroughTransform(Transform):
    def forward(self, X, context=None):
        return X, torch.zeros_like(X[:, 0])

    def inverse(self, X, context=None):
        return X, torch.zeros_like(X[:, 0])


class Custom2DHyperbolaTransform(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, X, context=None):
        XX = torch.zeros_like(X)
        XX[:, 0] = -X[:, 0]
        XX[:, 1] = -3.0 / (torch.abs(X[:, 0]) ** 0.8) + X[:, 1]
        return XX, torch.zeros_like(XX[:, 1])

    def inverse(self, X, context=None):
        XX = torch.zeros_like(X)
        XX[:, 0] = -X[:, 0]
        XX[:, 1] = 3.0 / (torch.abs(X[:, 0]) ** 0.8) + X[:, 1]
        return XX, torch.zeros_like(XX[:, 1])


class SinArcSinhTransform(Transform):
    def __init__(self, e, d):
        super().__init__()
        assert isinstance(e, int) or isinstance(e, float)
        assert isinstance(d, int) or isinstance(d, float)
        self.epsilon = e
        self.delta = d

    def _sas(self, x, epsilon, delta):
        return torch.sinh((torch.arcsinh(x) + epsilon) / delta)

    def _isas(self, x, epsilon, delta):
        return torch.sinh(delta * torch.arcsinh(x) - epsilon)

    def _ldisas(self, x, epsilon, delta):
        return torch.log(
            torch.abs(
                delta
                * torch.cosh(epsilon - delta * torch.arcsinh(x))
                / torch.sqrt(1 + x**2)
            )
        )

    def forward(self, X, context=None):
        XX = self._sas(X, self.epsilon, self.delta)
        ld = -self._ldisas(XX, self.epsilon, self.delta)
        ld = ld.flatten()
        return XX, ld

    def inverse(self, X, context=None):
        ld = self._ldisas(X, self.epsilon, self.delta)
        ld = ld.flatten()
        return self._isas(X, self.epsilon, self.delta), ld


class SAS2DTransform(Transform):
    def __init__(self, e=[0, 0], d=[1, 1]):
        super().__init__()
        self.epsilon = e
        self.delta = d
        self.t1 = SinArcSinhTransform(e[0], d[0])
        self.t2 = SinArcSinhTransform(e[1], d[1])

    def forward(self, X, context=None):
        TX = torch.zeros_like(X)
        ld = torch.zeros_like(X)
        TX[:, 0], ld[:, 0] = self.t1.forward(X[:, 0])
        TX[:, 1], ld[:, 1] = self.t2.forward(X[:, 1])
        return TX, ld.sum(axis=-1)

    def inverse(self, X, context=None):
        TX = torch.zeros_like(X)
        ld = torch.zeros_like(X)
        TX[:, 0], ld[:, 0] = self.t1.inverse(X[:, 0])
        TX[:, 1], ld[:, 1] = self.t2.inverse(X[:, 1])
        return TX, ld.sum(axis=-1)


class NaiveGaussianTransform(Transform):
    def __init__(self, inputs, weights=None):
        super().__init__()
        self._dim = inputs.shape[1]
        self._shift = torch.zeros(self._dim)
        if weights == None:
            n = inputs.shape[0]
            weights = torch.full([n], 1.0 / n)
        for d in range(self._dim):
            self._shift[d] = (
                inputs[:, d] @ weights
            ) / weights.sum()  # torch.mean(inputs[:,d])
        # fit covariance to inputs, then decompose
        if self._dim == 1:
            # 1D
            # std = torch.std(inputs,unbiased=True)
            cov = torch.Tensor(
                np.cov(
                    inputs.detach().numpy().flatten(), aweights=weights.detach().numpy()
                )
            )  # workaround for torch 1.9.1
            std = cov**0.5
            self._t = L1DTransform(std)
        else:
            # >1D
            # cov = torch.cov(inputs) # not supported in torch 1.9.1
            if inputs.shape[0] <= 1:
                print("WARNING: cov cannot be taken on a single sample.", inputs)
                L = torch.eye(self._dim)
                self._t = LTransform(L)
            else:
                cov = torch.Tensor(
                    np.cov(
                        inputs.detach().numpy(),
                        aweights=weights.detach().numpy(),
                        rowvar=False,
                    )
                )  # workaround for torch 1.9.1
                cov = torch.Tensor(make_pos_def(cov.detach().numpy()))
                L = safe_cholesky(cov)
                self._t = LTransform(L)

    def forward(self, inputs, context=None):
        x, ld = self._t.forward(inputs, context)
        x += self._shift
        return x, ld

    def inverse(self, inputs, context=None):
        x = inputs - self._shift
        return self._t.inverse(x, context)

class LTransform(Transform):
    def __init__(self, M=None, dim=None):
        super().__init__()
        if M is None:
            assert dim is not None
            self.L = torch.eye(dim)
        else:
            self.L = M
            self.Linv = torch.linalg.inv(M)
            self.ld = torch.logdet(M)

    def forward(self, X, context=None):
        return torch.matmul(X, self.L.T), torch.full([X.shape[0]], self.ld)

    def inverse(self, X, context=None):
        return torch.matmul(X, self.Linv.T), torch.full([X.shape[0]], -self.ld)


class L1DTransform(Transform):
    def __init__(self, M=None):
        super().__init__()
        if M is None:
            self.L = 1
            self.ld = 1
        else:
            self.L = M
            self.Linv = 1.0 / M
            self.ld = np.log(M)

    def forward(self, X, context=None):
        return X * self.L, torch.full([X.shape[0]], self.ld)

    def inverse(self, X, context=None):
        return X * self.Linv, torch.full([X.shape[0]], -self.ld)


class BetaMixtureMarginalTransform(Transform):
    def __init__(self, inputs, n_marg_beta_components=5):
        super().__init__()
        self.num_marginal_beta_components = n_marg_beta_components
        self.validfit = False
        self.ndim = inputs.shape[1]
        self._fit(inputs)

    def forward(self, X, context=None):
        n_k = X.shape[0]
        d = self.ndim
        if X.ndim == 1:
            X = X.reshape((n_k, d))
        U = torch.zeros_like(X)
        log_U = torch.zeros(n_k)
        for i in range(d):
            fitparams = self.marg_params[
                i
            ]  # needs to be univariate for scipy to fit it
            tmp = fitparams.cdf(t2n(X[:, i]))
            U[:, i] = n2t(tmp)
            log_U[:] += n2t(fitparams.logpdf(t2n(X[:, i])))
        return U, log_U

    def inverse(self, U, context=None):
        n_k = U.shape[0]
        d = self.ndim
        if U.ndim == 1:
            n_k = 1
            U = U.reshape((n_k, d))
        X = torch.zeros_like(U)
        log_X = torch.zeros(n_k)
        for i in range(d):
            fitparams = self.marg_params[
                i
            ]  # needs to be univariate for scipy to fit it
            X[:, i] = n2t(fitparams.ppf(t2n(U[:, i])))
            log_X[:] += n2t(-fitparams.logpdf(t2n(X[:, i])))
        return X, log_X

    def _fit(self, inputs):
        global PLOT_PROGRESS
        # each of the below is a list of lists of arrays or tuples
        n = inputs.shape[0]
        self.marg_params = []
        if n < max(self.ndim, 20) or self.ndim == 0:
            self.validfit = False
            for i in range(self.ndim):
                self.marg_params.append(
                    BetaMix(None, self.num_marginal_beta_components)
                )
        else:
            self.validfit = True
            for i in range(self.ndim):
                self.marg_params.append(
                    BetaMix(t2n(inputs[:, i]), self.num_marginal_beta_components)
                )  # needs to be univariate for scipy to fit it


class MaskedFixedNorm(Transform):
    def __init__(self, inputs, weights, mask, context_transform):
        super().__init__()
        self._dim = inputs.shape[1]
        self._scale = torch.ones(self._dim)
        self._shift = torch.zeros(self._dim)
        self._ct = context_transform  # TODO assert this is a lambda and equals mask given context
        if weights == None:
            n = inputs.shape[0]
            weights = torch.full([n], 1.0 / n)
        for d in range(self._dim):
            if mask[:, d].sum() == 0:
                self._shift[d] = 0
                self._scale[d] = 1
                continue
            self._shift[d] = (inputs[mask[:, d], d] @ weights[mask[:, d]]) / weights[
                mask[:, d]
            ].sum()  # torch.mean(inputs[:,d])
            try:
                cov = torch.Tensor(
                    np.cov(
                        inputs[mask[:, d], d].detach().numpy().flatten(),
                        aweights=weights[mask[:, d]].detach().numpy(),
                    )
                )  # workaround for torch 1.9.1
            except Exception:
                print(d)
                print(mask[:, d].sum())
                print(mask.shape)
                print(mask.sum())
                print(weights[mask[:, d]])
                sys.exit(0)
            self._scale[d] = cov**0.5
            self._scale[d] = max(self._scale[d], 1e-10)
            if torch.any(~torch.isfinite(self._scale)):
                print("inputs are singular, ", d)
                print(inputs[~torch.isfinite(inputs[mask[:, d], d])])
                print(self._scale)
                print(inputs)
                print(cov)
                print(weights)
                sys.exit(0)

    def inverse(self, inputs, context=None):
        if context is None:
            outputs = self._scale * inputs + self._shift
            return outputs, torch.log(
                torch.abs(self._scale * torch.ones_like(inputs))
            ).sum(axis=-1)
        else:
            mask = self._ct(context)
            N = inputs.shape[0]
            outputs = inputs.clone()
            outputs[mask] = (
                torch.tile(self._scale, (N, 1)) * inputs
                + torch.tile(self._shift, (N, 1))
            )[mask]
            return outputs, (
                torch.log(torch.abs(torch.tile(self._scale, (N, 1))))
                * mask.type(torch.int32)
            ).sum(axis=-1)

    def forward(self, inputs, context=None):
        if context is None:
            outputs = 1.0 / self._scale * (inputs - self._shift)
            return outputs, -torch.log(
                torch.abs(self._scale * torch.ones_like(inputs))
            ).sum(axis=-1)
        else:
            N = inputs.shape[0]
            mask = self._ct(context)
            outputs = inputs.clone()
            outputs[mask] = (
                torch.tile(self._scale ** (-1), (N, 1))
                * (inputs - torch.tile(self._shift, (N, 1)))
            )[mask]
            ld = (
                -torch.log(torch.abs(torch.tile(self._scale, (N, 1))))
                * mask.type(torch.int32)
            ).sum(axis=-1)
            return outputs, ld


class FixedLinear(Transform):
    def __init__(self, shift, scale):
        super().__init__()
        self._shift = shift
        self._scale = scale

    def inverse(self, inputs, context=None):
        outputs = 1.0 / self._scale * (inputs - self._shift)
        logabsdet = torchutils.sum_except_batch(
            -torch.log(torch.abs(self._scale * torch.ones_like(inputs))),
            num_batch_dims=1,
        )
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        outputs = self._scale * inputs + self._shift
        ld_temp = torch.log(torch.abs(self._scale * torch.ones_like(inputs)))
        logabsdet = torchutils.sum_except_batch(
            torch.log(torch.abs(self._scale * torch.ones_like(inputs))),
            num_batch_dims=1,
        )
        return outputs, logabsdet


class ConditionalMaskedTransform(Transform):
    def __init__(self, tf, context_transform):
        super().__init__()
        self._tf = tf  # todo assert transform
        self._ct = context_transform  # TODO assert this is a lambda and equals mask given context

    def forward(self, inputs, context=None):
        if context is None:
            return self._tf.forward(inputs)
        else:
            N = inputs.shape[0]
            mask = self._ct(context)
            outputs = inputs.clone()
            ld = torch.zeros_like(outputs)
            maskN = mask.sum()
            outputs_temp, ld_temp = self._tf.forward(inputs[mask].reshape((maskN, 1)))
            outputs[mask] = outputs_temp.reshape(maskN)
            ld[mask] = ld_temp.reshape(maskN)
            logabsdet = torchutils.sum_except_batch(ld, num_batch_dims=1)
            return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if context is None:
            return self._tf.inverse(inputs)
        else:
            N = inputs.shape[0]
            mask = self._ct(context)
            maskN = mask.sum()
            outputs = inputs.clone()
            ld = torch.zeros_like(outputs)
            outputs_temp, ld_temp = self._tf.inverse(inputs[mask].reshape((maskN, 1)))
            outputs[mask] = outputs_temp.reshape(maskN)
            ld[mask] = ld_temp.reshape(maskN)
            logabsdet = torchutils.sum_except_batch(ld, num_batch_dims=1)
            return outputs, logabsdet


class FixedNorm(Transform):
    def __init__(self, inputs, weights=None):
        super().__init__()
        self._dim = inputs.shape[1]
        self._scale = torch.ones(self._dim)
        self._shift = torch.zeros(self._dim)
        if weights is None:
            n = inputs.shape[0]
            weights = torch.full([n], 1.0 / n)
        for d in range(self._dim):
            self._shift[d] = (
                inputs[:, d] @ weights
            ) / weights.sum()  # torch.mean(inputs[:,d])
            cov = torch.Tensor(
                np.cov(
                    inputs[:, d].detach().numpy().flatten(),
                    aweights=weights.detach().numpy(),
                )
            )  # workaround for torch 1.9.1
            self._scale[d] = cov**0.5
            self._scale[d] = max(self._scale[d], 1e-10)
            if torch.any(~torch.isfinite(self._scale)):
                print("inputs are singular, ", d)
                print(inputs[~torch.isfinite(inputs[:, d])])
                print(self._scale)
                print(inputs)
                print(cov)
                print(weights)
                sys.exit(0)

    def inverse(self, inputs, context=None):
        outputs = self._scale * inputs + self._shift
        return outputs, torch.log(torch.abs(self._scale * torch.ones_like(inputs))).sum(
            axis=-1
        )

    def forward(self, inputs, context=None):
        outputs = 1.0 / self._scale * (inputs - self._shift)
        return outputs, -torch.log(
            torch.abs(self._scale * torch.ones_like(inputs))
        ).sum(axis=-1)


class LinearSquish(Transform):
    def __init__(self, low, high):
        super().__init__()
        self._low = torch.Tensor(low)
        self._high = torch.Tensor(high)
        self._range = self._high - self._low

    def forward(self, inputs, context=None):
        outputs = (inputs - self._low) * (1.0 / self._range)
        logabsdet = -torchutils.sum_except_batch(torch.log(self._range))
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        outputs = (inputs * self._range) + self._low
        logabsdet = torchutils.sum_except_batch(torch.log(self._range))
        return outputs, logabsdet


class ColumnSpecificTransform(Transform):
    def __init__(self, spec={}):
        super().__init__()
        for col, t in spec.items():
            assert isinstance(t, Transform)
            assert isinstance(col, int)
        self._spec = spec

    def forward(self, inputs, context=None):
        outputs = inputs.clone()
        ld = torch.zeros(inputs.shape[0])
        for col, t in self._spec.items():
            v, ldtemp = t.forward(inputs[:, col].reshape((inputs.shape[0], 1)), context)
            outputs[:, col] = v.reshape((inputs.shape[0],))
            ld += ldtemp
        return outputs, ld

    def inverse(self, inputs, context=None):
        outputs = inputs.clone()
        ld = torch.zeros(inputs.shape[0])
        for col, t in self._spec.items():
            v, ldtemp = t.inverse(inputs[:, col].reshape((inputs.shape[0], 1)), context)
            outputs[:, col] = v.reshape((inputs.shape[0],))
            ld += ldtemp
        return outputs, ld


class LogTransform(Transform):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, context=None):
        if torch.min(inputs) <= 0:
            print("Inputs negative ", inputs[inputs <= 0].shape)
            raise InputOutsideDomain()
        inputs = torch.clamp(inputs, self.eps, None)

        outputs = torch.log(inputs)
        if inputs.ndimension() > 1:
            sumdims = list(range(1, inputs.ndimension()))
            logabsdet = torchutils.sum_except_batch(
                -torch.log(inputs), num_batch_dims=1
            )
        else:
            logabsdet = -torch.log(inputs)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs = torch.exp(inputs)
        if inputs.ndimension() > 1:
            sumdims = list(range(1, inputs.ndimension()))
            logabsdet = -torchutils.sum_except_batch(
                -torch.log(outputs), num_batch_dims=1
            )
        else:
            logabsdet = torch.log(outputs)
        return outputs, logabsdet



class RationalQuadraticFlowFA(Flow):

    @classmethod
    def factory(
        cls,
        inputs,
        base_dist=None,
        boxing_transform=n.transforms.IdentityTransform(),
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 3000  # * max(1,dim_multiplier)
        ss_size = 32

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = n.transforms.IdentityTransform()
            ittr = n.transforms.IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform


        x = torch.tensor(inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim, hidden_features=ndim * 32, num_blocks=2, num_bins=10
                )
            )

        _transform = n.transforms.CompositeTransform(
            [
                ittr,
                n.transforms.CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        xx, __ = _transform.forward(x)
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = (
            lambda epoch: 0.999
        )  
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        hloss = torch.zeros(num_iter)
        lastloss = 1e9
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 99) : i + 1].mean()
                    print(i,avgloss)
                    if i>0 and np.abs(lastloss - avgloss) < 0.05:
                        break
                    elif i>0:
                        lastloss = avgloss
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 99) : i + 1].mean()
                    print(i,avgloss)
                    if i>0 and np.abs(lastloss - avgloss) < 0.05:
                        break
                    elif i>0:
                        lastloss = avgloss

class RationalQuadraticFlowFAV(Flow):

    @classmethod
    def factory(
        cls,
        inputs,
        base_dist=None,
        boxing_transform=n.transforms.IdentityTransform(),
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 3000  # * max(1,dim_multiplier)
        ss_size = 32

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = n.transforms.IdentityTransform()
            ittr = n.transforms.IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        val_percent = 0.1
        def get_train_val_idx(N,val_percent):
            sn = N*val_percent
            interval = N/sn
            val_idx =  np.array(np.arange(sn)*interval + interval/2,dtype=int)
            train_idx = np.setdiff1d(np.arange(N),val_idx)
            return train_idx, val_idx


        N = inputs.shape[0]
        if input_weights is not None:
            train_idx, validate_idx = get_train_val_idx(N,val_percent)
            weight_sort_idx = np.argsort(input_weights)
            train_idx = weight_sort_idx[train_idx]
            validate_idx = weight_sort_idx[validate_idx]
            
            x_train = torch.tensor(inputs[train_idx],dtype=torch.float32)
            x_train_w = input_weights[train_idx]
            x_train_w = np.exp(np.log(x_train_w) - logsumexp(np.log(x_train_w)))
            x_validate = torch.tensor(inputs[validate_idx],dtype=torch.float32)
            x_validate_w = input_weights[validate_idx]
            x_validate_w = np.exp(np.log(x_validate_w) - logsumexp(np.log(x_validate_w)))
        else:
            train_idx, validate_idx = get_train_val_idx(N,val_percent)
            x_train = torch.tensor(inputs[train_idx], dtype=torch.float32)
            x_train_w = np.full(train_idx.shape[0],1./train_idx.shape[0])
            x_validate = torch.tensor(inputs[validate_idx],dtype=torch.float32)
            x_validate_w = np.full(validate_idx.shape[0],1./validate_idx.shape[0])

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim, hidden_features=ndim * 32, num_blocks=2, num_bins=10
                )
            )

        _transform = n.transforms.CompositeTransform(
            [
                ittr,
                n.transforms.CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = (
            lambda epoch: 0.995 if epoch < 200 else 0.9971
        )  
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        train_hloss = torch.zeros(num_iter)
        val_hloss = torch.zeros(num_iter)
        train_lastloss = float('inf')
        val_lastloss = float('inf')
        if True:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x_train.shape[0], ss_size, p=x_train_w)
                loss = -myflow.log_prob(inputs=x_train[ids]).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    # do validation here
                    val_ids = np.random.choice(x_validate.shape[0], ss_size, p=x_validate_w)
                    val_loss = -myflow.log_prob(inputs=x_validate[val_ids]).mean()
                    val_hloss[i] = val_loss # store for assessment
                    train_hloss[i] = loss # store for assessment
                    train_avgloss = train_hloss[max(0, i - 99) : i + 1].mean()
                    val_avgloss = val_hloss[max(0, i - 99) : i + 1].mean()
                    if (i) % 100 == 0:
                        print(i,train_avgloss,val_avgloss)
                        #if i>0 and val_lastloss - val_avgloss < 0.05:
                        if i>0 and val_lastloss - val_avgloss < 0:
                            print("Finished training")
                            print(i,train_avgloss,val_avgloss)
                            break
                        elif i>0:
                            val_lastloss = val_avgloss


class RationalQuadraticFlow2(Flow):

    @classmethod
    def factory(
        cls,
        inputs,
        base_dist=None,
        boxing_transform=n.transforms.IdentityTransform(),
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 1000  # * max(1,dim_multiplier)
        ss_size = 128

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = n.transforms.IdentityTransform()
            ittr = n.transforms.IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        x = torch.tensor(inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim, hidden_features=ndim * 32, num_blocks=2, num_bins=10
                )
            )

        _transform = n.transforms.CompositeTransform(
            [
                ittr,
                n.transforms.CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        xx, __ = _transform.forward(x)
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = (
            lambda epoch: 0.992 if epoch < 200 else 0.9971
        )  
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        hloss = torch.zeros(num_iter)
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 250 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 250 == 0 and (i) > 0:
                    ss_size *= 2



class RationalQuadraticFlow(Flow):

    @classmethod
    def factory(
        cls,
        inputs,
        base_dist=None,
        standardising_transform=n.transforms.IdentityTransform(),
        boxing_transform=n.transforms.IdentityTransform(),
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 2000  # * max(1,dim_multiplier)
        ss_size = int(2 ** np.ceil(np.log2(inputs.shape[0] * 0.05)))

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = n.transforms.IdentityTransform()
            sttr = n.transforms.IdentityTransform()
            ittr = n.transforms.IdentityTransform()
        else:
            utr = boxing_transform
            sttr = standardising_transform
            ittr = initial_transform


        x = torch.tensor(inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim, hidden_features=ndim * 32, num_blocks=2, num_bins=10
                )
            )

        _transform = n.transforms.CompositeTransform(
            [
                ittr,
                n.transforms.CompositeCDFTransform(
                    utr, CompositeTransform([sttr, CompositeTransform(transforms)])
                ),
            ]
        )
        xx, __ = _transform.forward(x)
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = lambda epoch: 0.992 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        hloss = torch.zeros(num_iter)
        lastloss = 1e9
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    if lastloss - avgloss < 0.5:
                        break
                    else:
                        lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    if lastloss - avgloss < 0.5:
                        break
                    else:
                        lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2


class CauchyCDF1D(Transform):
    def __init__(self, location=None, scale=None, features=None):
        super().__init__()

    def forward(self, inputs, context=None):
        outputs = (1 / np.pi) * torch.atan(inputs) + 0.5
        logabsdet = torchutils.sum_except_batch(
            -np.log(np.pi) - torch.log(1 + inputs**2), num_batch_dims=1
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        outputs = torch.tan(np.pi * (inputs - 0.5))
        logabsdet = -torchutils.sum_except_batch(
            -np.log(np.pi) - torch.log(1 + outputs**2), num_batch_dims=1
        )
        return outputs, logabsdet


from torch.distributions.studentT import StudentT


class StudentTDist(Distribution):
    """A multivariate t-distribution with zero mean and unit covariance."""

    def __init__(self, shape, df=2):
        super().__init__()
        self._shape = torch.Size(shape)
        self.register_buffer(
            "_df", torch.tensor(df, dtype=torch.float32), persistent=False
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return torchutils.sum_except_batch(
            StudentT(df=self._df).log_prob(inputs), num_batch_dims=1
        )

    def _sample(self, num_samples, context):
        if context is None:
            return StudentT(df=self._df).rsample(
                torch.Size((num_samples,)) + self._shape
            )
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            st = StudentT(
                df=self._df,
                loc=torch.tensor([0.0], device=context.device, dtype=torch.float64),
                scale=torch.tensor([1.0], device=context.device, dtype=torch.float64),
            )
            samples = st.rsample(
                torch.Size((context_size * num_samples,)) + self._shape
            )
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)


class RationalTailQuadraticFlow(Flow):

    @classmethod
    def factory(
        cls,
        inputs,
        base_dist=None,
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 2  # fix for now
        num_iter = 1000  # * max(1,dim_multiplier)
        ss_size = int(2 ** np.ceil(np.log2(inputs.shape[0] * 0.05)))

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            ittr = n.transforms.IdentityTransform()
        else:
            ittr = initial_transform

        x = torch.tensor(inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            if _ > 0:
                transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim,
                    hidden_features=ndim * 32,
                    num_blocks=2,
                    num_bins=16,
                    tails="linear",
                )
            )

        _transform = n.transforms.CompositeTransform(
            [ittr, CompositeTransform(transforms)]
        )
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=1e-3)
        lmbda = lambda epoch: 0.992 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        hloss = torch.zeros(num_iter)
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2


class ConditionalRationalQuadraticFlow(Flow):

    @classmethod
    def factory(
        cls,
        inputs,
        context_inputs,
        base_dist=None,
        boxing_transform=n.transforms.IdentityTransform(),
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]
        ncdim = context_inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 500  # * max(1,dim_multiplier)
        if input_weights is not None:
            logw = np.log(input_weights)
            logw -= logsumexp(logw)
            ess = np.exp(-logsumexp(2 * logw))
            ss_size = int(2 ** np.ceil(np.log2(ess * 0.05)))
        else:
            ss_size = int(2 ** np.ceil(np.log2(inputs.shape[0] * 0.05)))

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = n.transforms.IdentityTransform()
            ittr = n.transforms.IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(context_inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim,
                    hidden_features=ndim * 32,
                    context_features=ncdim,
                    num_blocks=2,
                    num_bins=10,
                )
            )

        _transform = n.transforms.CompositeTransform(
            [
                ittr,
                n.transforms.CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = lambda epoch: 0.992 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        hloss = torch.zeros(num_iter)
        lastloss = 1e9
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                loss = -myflow.log_prob(inputs=x[ids], context=y[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                loss = -myflow.log_prob(inputs=x[ids], context=y[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2

        if PLOT_PROGRESS:
            import matplotlib.pyplot as plt

            if x.shape[1] == 2:
                f, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), s=0.1)
                yy, yldet = myflow._transform.forward(x)
                y = yy.detach().numpy()
                ax[1].scatter(y[:, 0], y[:, 1], s=0.1, color="yellow")
                plt.show()
            elif x.shape[1] == 1:
                f, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].hist(x[:, 0].detach().numpy(), bins=50)
                yy, yldet = myflow._transform.forward(x)
                y = yy.detach().numpy()
                ax[1].hist(y[:, 0], bins=50, color="yellow")
                plt.show()
        return myflow


class ConditionalMaskedRationalQuadraticFlow(Flow):

    @classmethod
    def factory(
        cls,
        inputs,
        context_inputs,
        context_mask,
        aux_dist=StandardNormal((1,)),
        base_dist=None,
        boxing_transform=n.transforms.IdentityTransform(),
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]
        ncdim = context_inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 1000  # * max(1,dim_multiplier)
        ss_size = 128

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = n.transforms.IdentityTransform()
            ittr = n.transforms.IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        x = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(context_inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim,
                    hidden_features=ndim * 32,
                    context_features=ncdim,
                    num_blocks=2,
                    num_bins=10,
                )
            )

        _transform = n.transforms.CompositeTransform(
            [
                ittr,
                n.transforms.CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = lambda epoch: 0.992 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        hloss = torch.zeros(num_iter)
        lastloss = 1e9
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                x_m = torch.zeros_like(x[ids])
                x_m[~context_mask[ids]] = x[ids][~context_mask[ids]]
                x_m[context_mask[ids]] = aux_dist.sample(
                    (int(context_mask[ids].sum()),)
                ).flatten()
                loss = -myflow.log_prob(inputs=x_m, context=y[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    if lastloss - avgloss < 0.01:
                        break
                    lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                x_m = torch.zeros_like(x[ids])
                x_m[~context_mask[ids]] = x[ids][~context_mask[ids]]
                x_m[context_mask[ids]] = aux_dist.sample(
                    (int(context_mask[ids].sum()),)
                ).flatten()
                loss = -myflow.log_prob(inputs=x_m, context=y[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    avgloss = hloss[max(0, i - 50) : i + 1].mean()
                    print(i, avgloss)
                    if lastloss - avgloss < 0.01:
                        break
                    lastloss = avgloss
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2

        if PLOT_PROGRESS:
            import matplotlib.pyplot as plt

            if x.shape[1] == 2:
                f, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].scatter(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), s=0.1)
                yy, yldet = myflow._transform.forward(x)
                y = yy.detach().numpy()
                ax[1].scatter(y[:, 0], y[:, 1], s=0.1, color="yellow")
                plt.show()
            elif x.shape[1] == 1:
                f, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].hist(x[:, 0].detach().numpy(), bins=50)
                yy, yldet = myflow._transform.forward(x)
                y = yy.detach().numpy()
                ax[1].hist(y[:, 0], bins=50, color="yellow")
                plt.show()
        return myflow


class RationalQuadraticFlowProposal(Flow):

    @classmethod
    def factory(
        cls,
        drawBirth,
        inputs,
        base_dist=None,
        boxing_transform=n.transforms.IdentityTransform(),
        initial_transform=n.transforms.IdentityTransform(),
        input_weights=None,
    ):
        global PLOT_PROGRESS

        ndim = inputs.shape[1]

        # Configuration
        dim_multiplier = int(np.log2(1 + np.log2(ndim)))
        num_layers = 1 + dim_multiplier  # int(np.log2(ndim))
        num_layers = 3  # fix for now
        num_iter = 1000  # * max(1,dim_multiplier)
        ss_size = int(2 ** np.ceil(np.log2(inputs.shape[0] * 0.05)))

        if base_dist is None:
            base_dist = Uniform(low=torch.zeros(ndim), high=torch.ones(ndim))
            utr = n.transforms.IdentityTransform()
            ittr = n.transforms.IdentityTransform()
        else:
            utr = boxing_transform
            ittr = initial_transform

        x = torch.tensor(inputs, dtype=torch.float32)

        transforms = []

        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=ndim))
            transforms.append(
                n.transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=ndim, hidden_features=ndim * 32, num_blocks=2, num_bins=10
                )
            )

        _transform = n.transforms.CompositeTransform(
            [
                ittr,
                n.transforms.CompositeCDFTransform(utr, CompositeTransform(transforms)),
            ]
        )
        xx, __ = _transform.forward(x)
        myflow = cls(_transform, base_dist)
        optimizer = optim.Adam(myflow.parameters(), lr=3e-3)
        lmbda = lambda epoch: 0.992 if epoch < 200 else 0.9971
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lr_lambda=lmbda
        )
        hloss = torch.zeros(num_iter)
        if input_weights is None:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2
        else:
            for i in range(num_iter):
                optimizer.zero_grad()
                ids = np.random.choice(x.shape[0], ss_size, p=input_weights)
                loss = -myflow.log_prob(inputs=x[ids]).mean()
                with torch.no_grad():
                    hloss[i] = loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (i) % 100 == 0:
                    # print(i, loss)
                    print(i, hloss[max(0, i - 50) : i + 1].mean())
                if (i) % 200 == 0 and (i) > 0:
                    ss_size *= 2


if __name__ == "__main__":
    pass
