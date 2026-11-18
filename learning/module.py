import torch.nn as nn


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        if tsteps == 100:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 16, output_size), self.activation_fn()
            )
        elif tsteps == 400:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 91, output_size), self.activation_fn(),
            )
        elif tsteps == 10:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 20:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        else:
            raise NotImplementedError()


    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        # (bs, T, channels) -> (bs, channels, T) for Conv1d. A plain reshape to (bs, -1, T) would mix
        # time steps and channels instead of transposing them.
        output = self.conv_layers(projection.reshape([bs, T, -1]).permute(0, 2, 1))
        output = self.linear_output(output)
        return output


class MLPHistoryEncoder(nn.Module):
    """Phase 2 encoder: flattened history of [x_k, a_{k-1}], k = t-history_len ... t-1 (oldest first)
    -> ẑ, an estimate of the Phase 1 latent z = env_net(e). Tanh output, like z."""

    def __init__(self, step_size, history_len, latent_size, hidden=(256, 128)):
        super(MLPHistoryEncoder, self).__init__()
        layers, size = [], step_size * history_len
        for width in hidden:
            layers += [nn.Linear(size, width), nn.Tanh()]
            size = width
        layers += [nn.Linear(size, latent_size), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, history):
        return self.net(history)


def make_history_encoder(cfg):
    """History encoder from hyperparam.yaml (dagger.encoder). Only the MLP exists for now; CNN (e.g. the
    StateHistoryEncoder above) and RNN versions plug in here."""
    space, enc = cfg["observation_space"], cfg["dagger"]["encoder"]
    step_size = int(space["state_obs_size"]) + int(space["act_size"])
    if enc["type"] == "mlp":
        return MLPHistoryEncoder(step_size, int(space["history_len"]), int(space["latent_size"]),
                                 [int(w) for w in enc["hidden"]])
    raise NotImplementedError("history encoder type '{}' is not implemented yet".format(enc["type"]))
