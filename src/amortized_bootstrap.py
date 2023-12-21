import torch


class AmortizedBootstrapLinreg():
    def __init__(self, generator):
        self.generator = generator
        self.latent_size = generator[0].in_features

    def sample(self, k=1):
        z = torch.randn((k, self.latent_size))
        theta_pred = self.generator(z).detach().numpy()
        return theta_pred