import torch


class Adam():
    def __init__(self, state_dict, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.m_dw = {param_name: torch.zeros_like(state_dict[param_name]) for param_name in state_dict}
        self.v_dw = {param_name: torch.zeros_like(state_dict[param_name]) for param_name in state_dict}

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.t = 1

    def update(self, w, dw):
        m_dw_corr = {param_name: torch.zeros_like(self.m_dw[param_name]) for param_name in self.m_dw}
        v_dw_corr = {param_name: torch.zeros_like(self.m_dw[param_name]) for param_name in self.m_dw}

        for param_name in self.m_dw:
            # momentum + RMS
            self.m_dw[param_name] = self.beta1 * self.m_dw[param_name] + (1 - self.beta1) * dw[param_name]
            self.v_dw[param_name] = self.beta2 * self.v_dw[param_name] + (1 - self.beta2) * (dw[param_name] ** 2)
            # bias correction
            m_dw_corr[param_name] = self.m_dw[param_name] / (1 - self.beta1 ** self.t)
            v_dw_corr[param_name] = self.v_dw[param_name] / (1 - self.beta2 ** self.t)

        # update
        for param_name in self.m_dw:
            w[param_name] = w[param_name] - self.lr * (
                        m_dw_corr[param_name] / (torch.sqrt(v_dw_corr[param_name]) + self.epsilon))

        self.t += 1

        return w
