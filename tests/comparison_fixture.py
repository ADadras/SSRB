"""Synthetic inputs only; expected outputs were captured from the authors' code."""
import torch
from torch.distributions import Independent, Normal


def probunet_case(model_class, dice_loss, kl_normal):
    torch.manual_seed(719)
    model=model_class(base=2,latent_dim=3).eval()
    image=torch.linspace(-1,1,2*3*32*32).reshape(2,3,32,32)
    target=(image[:,:1]>.1).float()
    pm,pl,z,skips=model.encode_prior(image)
    qm,ql=model.encode_posterior(image,target)
    logits=model.decode(z,skips,qm)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)+dice_loss(logits,target)+.01*kl_normal(qm,ql,pm,pl)
    loss.backward()
    return dict(prior_mu=pm.detach().tolist(),posterior_mu=qm.detach().tolist(),
                logits=logits[0,0,:3,:4].detach().tolist(),loss=float(loss.detach()),
                gradient=model.fuse.weight.grad.flatten()[:12].tolist())


class Ambiguity(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.mean=torch.nn.Parameter(torch.tensor(offset))
        self.log_scale=torch.nn.Parameter(torch.tensor(.1))
    def forward(self, image, target):
        mean=target.mean((1,2,3))[:,None].expand(-1,3)+self.mean
        scale=self.log_scale.exp().expand_as(mean)
        return Independent(Normal(mean,scale),1)


class Components(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Conv2d(5,2,1)
        self.ambiguity_control=Ambiguity(.2)
        self.ambiguity_modelling=Ambiguity(-.1)


class Diffusion:
    def q_sample(self, clean, timestep, noise):
        return .75*clean+.25*noise
    def _scale_timesteps(self,t):
        return t
    def _vb_terms_bpd(self,model,x_start,x_t,t,clip_denoised):
        return {'output':model(x_t,t).square().flatten(1).mean(1)}


class NoiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv2d(5,2,1)
    def forward(self,value,timestep):
        return self.conv(value)


def cimd_case(training_objective):
    torch.manual_seed(829)
    components=Components();components.model=NoiseModel()
    image=torch.linspace(-1,1,2*4*8*8).reshape(2,4,8,8)
    target=(image[:,:1]>0).float();before=target.clone()
    noise=torch.linspace(-2,2,128).reshape(2,1,8,8)
    loss,terms=training_objective(components,Diffusion(),image,target,torch.tensor([12,24]),noise=noise)
    assert torch.equal(target,before)
    loss.backward()
    return dict(loss=float(loss.detach()),terms={k:v.detach().tolist() for k,v in terms.items()},
                model_gradient=components.model.conv.weight.grad.flatten().tolist(),
                control_gradient=float(components.ambiguity_control.mean.grad),
                modelling_gradient=float(components.ambiguity_modelling.mean.grad))
