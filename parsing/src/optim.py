import torch

from parsing.src.constants import *


class AdamOptim:
    def __init__(self, enc_state_dict, dec_state_dict, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.enc_m_dw = {param_name: torch.zeros_like(enc_state_dict[param_name]) for param_name in enc_state_dict
                     if param_name not in FROZEN_EMB_LAYERS_NAME}
        self.enc_v_dw = {param_name: torch.zeros_like(enc_state_dict[param_name]) for param_name in enc_state_dict
                     if param_name not in FROZEN_EMB_LAYERS_NAME}

        if dec_state_dict is not None:
            self.dec_m_dw = {param_name: torch.zeros_like(dec_state_dict[param_name]) for param_name in dec_state_dict
                         if param_name not in FROZEN_EMB_LAYERS_NAME}
            self.dec_v_dw = {param_name: torch.zeros_like(dec_state_dict[param_name]) for param_name in dec_state_dict
                         if param_name not in FROZEN_EMB_LAYERS_NAME}

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.t = 1

    def update(self, enc_w, enc_dw, dec_w=None, dec_dw=None, batch_w_i=None):

        # encoder update #
        # momentum + RMS
        for param_name in self.enc_m_dw:
            if param_name == ACTIVE_EMB_LAYER_NAME:
                self.enc_m_dw[param_name][batch_w_i] = self.beta1 * self.enc_m_dw[param_name][batch_w_i] + (1 - self.beta1) * enc_dw[param_name]
                self.enc_v_dw[param_name][batch_w_i] = self.beta2 * self.enc_v_dw[param_name][batch_w_i] + (1 - self.beta2) * (enc_dw[param_name] ** 2)
            else:
                self.enc_m_dw[param_name] = self.beta1*self.enc_m_dw[param_name] + (1-self.beta1)*enc_dw[param_name]
                self.enc_v_dw[param_name] = self.beta2*self.enc_v_dw[param_name] + (1-self.beta2)*(enc_dw[param_name]**2)

        # bias correction
        m_dw_corr = {param_name: torch.zeros_like(enc_dw[param_name]) for param_name in enc_dw}
        v_dw_corr = {param_name: torch.zeros_like(enc_dw[param_name]) for param_name in enc_dw}
        for param_name in self.enc_m_dw:
            if param_name == ACTIVE_EMB_LAYER_NAME:
                m_dw_corr[param_name] = self.enc_m_dw[param_name][batch_w_i] / (1 - self.beta1 ** self.t)
                v_dw_corr[param_name] = self.enc_v_dw[param_name][batch_w_i] / (1 - self.beta2 ** self.t)
            else:
                m_dw_corr[param_name] = self.enc_m_dw[param_name]/(1-self.beta1**self.t)
                v_dw_corr[param_name] = self.enc_v_dw[param_name]/(1-self.beta2**self.t)

        # update
        for param_name in self.enc_m_dw:
            if param_name == ACTIVE_EMB_LAYER_NAME:
                enc_w[param_name][batch_w_i] = enc_w[param_name][batch_w_i] - self.lr * (m_dw_corr[param_name] / (torch.sqrt(v_dw_corr[param_name]) + self.epsilon))
            else:
                enc_w[param_name] = enc_w[param_name] - self.lr * (m_dw_corr[param_name] / (torch.sqrt(v_dw_corr[param_name]) + self.epsilon))

        if dec_w is not None and dec_dw is not None:
            # decoder update #
            # momentum + RMS
            for param_name in self.dec_m_dw:
                self.dec_m_dw[param_name] = self.beta1*self.dec_m_dw[param_name] + (1-self.beta1)*dec_dw[param_name]
                self.dec_v_dw[param_name] = self.beta2*self.dec_v_dw[param_name] + (1-self.beta2)*(dec_dw[param_name]**2)

            # bias correction
            m_dw_corr = {param_name: torch.zeros_like(dec_dw[param_name]) for param_name in dec_dw}
            v_dw_corr = {param_name: torch.zeros_like(dec_dw[param_name]) for param_name in dec_dw}
            for param_name in self.dec_m_dw:
                m_dw_corr[param_name] = self.dec_m_dw[param_name]/(1-self.beta1**self.t)
                v_dw_corr[param_name] = self.dec_v_dw[param_name]/(1-self.beta2**self.t)

            # update
            for param_name in self.dec_m_dw:
                dec_w[param_name] = dec_w[param_name] - self.lr * (m_dw_corr[param_name] / (torch.sqrt(v_dw_corr[param_name]) + self.epsilon))

            self.t += 1

            return enc_w, dec_w

        self.t += 1

        return enc_w





