import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FullyConnected(nn.Sequential):
    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)


class DistributionNet(nn.Module):
    @staticmethod
    def get_class(dtype):
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))


class BernoulliNet(DistributionNet):
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return logits,

    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)


class NormalNet(DistributionNet):
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)


class DiagNormalNet(nn.Module):
    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., :self.dim].clamp(min=-1e2, max=1e2)
        scale = nn.functional.softplus(loc_scale[..., self.dim:]).add(1e-3).clamp(max=1e2)
        return loc, scale


class PreWhitener(nn.Module):
    def __init__(self, data):
        super().__init__()
        with torch.no_grad():
            loc = data.mean(0)
            scale = data.std(0)
            scale[~(scale > 0)] = 1.0
            self.register_buffer("loc", loc)
            self.register_buffer("inv_scale", scale.reciprocal())

    def forward(self, data):
        return (data - self.loc) * self.inv_scale


class Guide(PyroModule):
    def __init__(self, config):
        self.latent_Ztm_dim = config["latent_Ztm_dim"]
        self.latent_Zty_dim = config["latent_Zty_dim"]
        self.latent_Zmy_dim = config["latent_Zmy_dim"]

        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        super().__init__()
        self.t_nn = BernoulliNet([config["latent_Ztm_dim"] + config["latent_Zty_dim"]])

        self.m_nn = FullyConnected([config["latent_Ztm_dim"] + config["latent_Zmy_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())

        self.m0_nn = OutcomeNet([config["hidden_dim"]])
        self.m1_nn = OutcomeNet([config["hidden_dim"]])

        self.y_nn = FullyConnected([1 + config["latent_Zty_dim"] + config["latent_Zmy_dim"]] +
                                   [config["hidden_dim"]] * (config["num_layers"] - 1),
                                   final_activation=nn.ELU())
        self.y0_nn = OutcomeNet([config["hidden_dim"]])
        self.y1_nn = OutcomeNet([config["hidden_dim"]])

        self.ztm_nn = FullyConnected(
            [config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )
        self.ztm0_nn = DiagNormalNet([config["hidden_dim"], config["latent_Ztm_dim"]])

        self.zty_nn = FullyConnected(
            [config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )
        self.zty0_nn = DiagNormalNet([config["hidden_dim"], config["latent_Zty_dim"]])

        self.zmy_nn = FullyConnected(
            [config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )
        self.zmy0_nn = DiagNormalNet([config["hidden_dim"], config["latent_Zmy_dim"]])


    def forward(self, x, m=None, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            zmy = pyro.sample("zmy", self.zmy_dist(x))
            ztm = pyro.sample("ztm", self.ztm_dist(x))
            zty = pyro.sample("zty", self.zty_dist(x))
            t = pyro.sample("t", self.t_dist(ztm, zty), obs=t, infer={"is_auxiliary": True})
            m = pyro.sample("m", self.m_dist(t, zmy, ztm), obs=m, infer={"is_auxiliary": True})
            y = pyro.sample("y", self.y_dist(t, m, zmy, zty), obs=y, infer={"is_auxiliary": True})


    def t_dist(self, ztm, zty):
        ztm_zty = torch.cat([ztm, zty], dim=-1)
        logits, = self.t_nn(ztm_zty)
        return dist.Bernoulli(logits=logits)

    def m_dist(self, t, zmy, ztm):
        zmy_ztm = torch.cat([zmy, ztm], dim=-1)
        hidden = self.m_nn(zmy_ztm)
        params0 = self.m0_nn(hidden)
        params1 = self.m1_nn(hidden)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.m0_nn.make_dist(*params)

    def y_dist(self, t, m, zmy, zty):
        m_zmy = torch.cat([m.unsqueeze(-1), zmy], dim=-1)
        m_zmy_zty = torch.cat([m_zmy, zty], dim=-1)
        hidden = self.y_nn(m_zmy_zty)
        params0 = self.y0_nn(hidden)
        params1 = self.y1_nn(hidden)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)


    def ztm_dist(self, x):
        hidden = self.ztm_nn(x)
        params0 = self.ztm0_nn(hidden)
        return dist.Normal(*params0).to_event(1)

    def zty_dist(self, x):
        hidden = self.zty_nn(x)
        params0 = self.zty0_nn(hidden)
        return dist.Normal(*params0).to_event(1)

    def zmy_dist(self, x):
        hidden = self.zmy_nn(x)
        params0 = self.zmy0_nn(hidden)
        return dist.Normal(*params0).to_event(1)


class TraceCausalEffect_ELBO(Trace_ELBO):
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        blocked_names = [name for name, site in guide_trace.nodes.items()
                         if site["type"] == "sample" and site["is_observed"]]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace)

        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            loss = loss - torch_item(log_q)
            surrogate_loss = surrogate_loss - log_q

        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))


class Model(PyroModule):
    def __init__(self, config):
        self.latent_Ztm_dim = config["latent_Ztm_dim"]
        self.latent_Zty_dim = config["latent_Zty_dim"]
        self.latent_Zmy_dim = config["latent_Zmy_dim"]

        super().__init__()
        self.x_nn = DiagNormalNet(
            [config["latent_Ztm_dim"] + config["latent_Zty_dim"] + config["latent_Zmy_dim"]]
            + [config["hidden_dim"]] * config["num_layers"]
            + [config["feature_dim"]]
        )
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        self.m0_nn = OutcomeNet(
            [config["latent_Ztm_dim"] + config["latent_Zmy_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.m1_nn = OutcomeNet(
            [config["latent_Ztm_dim"] + config["latent_Zmy_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )

        self.y0_nn = OutcomeNet(
            [config["latent_Zty_dim"] + config["latent_Zmy_dim"] + 1] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y1_nn = OutcomeNet(
            [config["latent_Zty_dim"] + config["latent_Zmy_dim"] + 1] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.t_nn = BernoulliNet([config["latent_Ztm_dim"] + config["latent_Zty_dim"]])

    def forward(self, x, m=None, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            zmy = pyro.sample("zmy", self.zmy_dist())
            ztm = pyro.sample("ztm", self.ztm_dist())
            zty = pyro.sample("zty", self.zty_dist())
            t = pyro.sample("t", self.t_dist(ztm, zty), obs=t)
            x = pyro.sample("x", self.x_dist(zmy, ztm, zty), obs=x)
            m = pyro.sample("m", self.m_dist(t, zmy, ztm), obs=m)
            y = pyro.sample("y", self.y_dist(t, m, zmy, zty), obs=y)
        return y

    def y_mean(self, x, m, t=None):
        with pyro.plate("data", x.size(0)):
            zmy = pyro.sample("zmy", self.zmy_dist())
            ztm = pyro.sample("ztm", self.ztm_dist())
            zty = pyro.sample("zty", self.zty_dist())
            x = pyro.sample("x", self.x_dist(zmy, ztm, zty), obs=x)
            t = pyro.sample("t", self.t_dist(ztm, zty), obs=t)
            m = pyro.sample("m", self.m_dist(t, zmy, ztm), obs=m)
        return self.y_dist(t, m, zmy, zty).mean

    def m_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            zmy = pyro.sample("zmy", self.zmy_dist())
            ztm = pyro.sample("ztm", self.ztm_dist())
            zty = pyro.sample("zty", self.zty_dist())
            x = pyro.sample("x", self.x_dist(zmy, ztm, zty), obs=x)
            t = pyro.sample("t", self.t_dist(ztm, zty), obs=t)
        return self.m_dist(t, zmy, ztm).mean

    def zmy_dist(self):
        return dist.Normal(0, 1).expand([self.latent_Zmy_dim]).to_event(1)

    def ztm_dist(self):
        return dist.Normal(0, 1).expand([self.latent_Ztm_dim]).to_event(1)

    def zty_dist(self):
        return dist.Normal(0, 1).expand([self.latent_Zty_dim]).to_event(1)

    def x_dist(self, zmy, ztm, zty):
        zmy_ztm = torch.cat([zmy, ztm], dim=-1)
        zmy_ztm_zty = torch.cat([zty, zmy_ztm], dim=-1)
        loc, scale = self.x_nn(zmy_ztm_zty)
        return dist.Normal(loc, scale).to_event(1)

    def m_dist(self, t, zmy, ztm):
        zmy_ztm = torch.cat([zmy, ztm], dim=-1)
        params0 = self.m0_nn(zmy_ztm)
        params1 = self.m1_nn(zmy_ztm)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.m0_nn.make_dist(*params)

    def y_dist(self, t, m, zmy, zty):
        m_zmy = torch.cat([m.unsqueeze(-1), zmy], dim=-1)
        m_zmy_zty = torch.cat([m_zmy, zty], dim=-1)
        params0 = self.y0_nn(m_zmy_zty)
        params1 = self.y1_nn(m_zmy_zty)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    def t_dist(self, ztm, zty):
        ztm_zty = torch.cat([ztm, zty], dim=-1)
        logits, = self.t_nn(ztm_zty)
        return dist.Bernoulli(logits=logits)


class DMA_VAE(nn.Module):
    def __init__(self, feature_dim, outcome_dist="normal",
                 latent_Ztm_dim=0, latent_Zty_dim=0, latent_Zmy_dim=0, hidden_dim=0, num_layers=0, num_samples=0):
        config = dict(feature_dim=feature_dim, latent_Ztm_dim=latent_Ztm_dim, latent_Zty_dim=latent_Zty_dim, latent_Zmy_dim=latent_Zmy_dim,
                      hidden_dim=hidden_dim, num_layers=num_layers,
                      num_samples=num_samples)
        config["outcome_dist"] = outcome_dist
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        super().__init__()
        self.model = Model(config)
        self.guide = Guide(config)
        self.to(device)

    def fit(self, x, m, t, y,
            num_epochs=100,
            batch_size=100,
            learning_rate=1e-3,
            learning_rate_decay=0.1,
            weight_decay=1e-4):

        assert x.dim() == 2 and x.size(-1) == self.feature_dim
        assert m.shape == x.shape[:1]
        assert t.shape == x.shape[:1]
        assert y.shape == y.shape[:1]
        self.whiten = PreWhitener(x)

        dataset = TensorDataset(x, m, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logger.info("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam({"lr": learning_rate,
                             "weight_decay": weight_decay,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO())
        losses = []
        for epoch in range(num_epochs):
            for x, m, t, y in dataloader:
                x = self.whiten(x)
                loss = svi.step(x, m, t, y, size=len(dataset)) / len(dataset)
                logger.debug("step {: >5d} loss = {:0.6g}".format(len(losses), loss))
                assert not torch_isnan(loss)
                losses.append(loss)
            print("Epoch:" ,int(epoch))
        return losses

    @torch.no_grad()
    def effect_estimation(self, x, num_samples=None, batch_size=None):
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        logger.info("Evaluating {} minibatches".format(len(dataloader)))
        result_NDE = []
        result_NIEr = []
        result_NIE = []
        result_ATE = []
        for x in dataloader:
            x = self.whiten(x)
            with pyro.plate("num_particles", num_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=["m", "t", "y"]):
                    self.guide(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    m0 = poutine.replay(self.model.m_mean, tr.trace)(x)
                with poutine.do(data=dict(t=torch.ones(()))):
                    m1 = poutine.replay(self.model.m_mean, tr.trace)(x)
                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0_m0 = poutine.replay(self.model.y_mean, tr.trace)(x, m0)
                    y0_m1 = poutine.replay(self.model.y_mean, tr.trace)(x, m1)
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1_m0 = poutine.replay(self.model.y_mean, tr.trace)(x, m0)
                    y1_m1 = poutine.replay(self.model.y_mean, tr.trace)(x, m1)
                NDE = (y1_m0 - y0_m0).mean(0)
                NIEr = (y1_m1 - y1_m0).mean(0)
                NIE = (y0_m1 - y0_m0).mean(0)
                ATE = (y1_m1 - y0_m0).mean(0)
                if not torch._C._get_tracing_state():
                    logger.debug("batch ate = {:0.6g}".format(NDE.mean()))
                    logger.debug("batch ate = {:0.6g}".format(NIEr.mean()))
                    logger.debug("batch ate = {:0.6g}".format(NIE.mean()))
                    logger.debug("batch ate = {:0.6g}".format(ATE.mean()))
                result_NDE.append(NDE)
                result_NIEr.append(NIEr)
                result_NIE.append(NIE)
                result_ATE.append(ATE)
            return torch.cat(result_NDE), torch.cat(result_NIEr), torch.cat(result_NIE), torch.cat(result_ATE)
