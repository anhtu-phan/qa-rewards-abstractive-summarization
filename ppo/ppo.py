# https://github.com/lvwerra/trl/blob/master/trl/ppo.py

__all__ = ['AdaptiveKLController', 'FixedKLController', 'PPOTrainer']

# Cell
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random

from .core import (logprobs_from_logits,
                   whiten,
                   clip_by_value,
                   entropy_from_logits,
                   flatten_dict,
                   average_torch_dicts,
                   stats_to_np,
                   stack_dicts,
                   add_suffix)


# Cell


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


# Cell
class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


# Cell
class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """

    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef": 0.2,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": .1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
    }

    def __init__(self, model, ref_model, **ppo_params):
        """
        Initialize PPOTrainer.

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000

        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ref_model = ref_model
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])

        if self.ppo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                               self.ppo_params['target'],
                                               self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

    def step(self, query, response, scores, decoder_input_ids=None):
        """
        Run a PPO optimisation step.

        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.ppo_params['batch_size']
        timing = dict()
        t0 = time.time()

        gen_len = response.shape[1]
        model_input = query

        t = time.time()
        if decoder_input_ids is not None:
            logprobs, ref_logprobs, values = self.batched_forward_pass(model_input, gen_len,
                                                                       decode_inputs=decoder_input_ids)
        else:
            logprobs, ref_logprobs, values = self.batched_forward_pass(model_input, gen_len)

        timing['time/ppo/forward_pass'] = time.time() - t

        t = time.time()
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time() - t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.ppo_params['ppo_epochs']):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                if decoder_input_ids is None:
                    train_stats = self.train_minibatch(logprobs[idx:idx + 1], values[idx:idx + 1],
                                                       rewards[idx:idx + 1], query[idx:idx + 1],
                                                       response[idx:idx + 1], model_input[idx:idx + 1])
                else:
                    train_stats = self.train_minibatch(logprobs[idx:idx + 1], values[idx:idx + 1],
                                                       rewards[idx:idx + 1], query[idx:idx + 1],
                                                       response[idx:idx + 1], model_input[idx:idx + 1],
                                                       decoder_input_ids[idx:idx + 1])
                all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time() - t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time() - t0
        stats.update(timing)
        return stats

    def batched_forward_pass(self, model_input, gen_len, decode_inputs=None):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        logprobs = []
        ref_logprobs = []
        values = []

        for i in range(int(bs / fbs)):
            m_input = model_input[i * fbs:(i + 1) * fbs]
            if decode_inputs is not None:
                m_decode_ids = decode_inputs[i * fbs:(i + 1) * fbs]
                logits, _, v = self.model(input_ids=m_input, decoder_input_ids=m_decode_ids)
                if not self.ppo_params['use_cuda_for_ref_model']:
                    m_input = m_input.cpu()
                    m_decode_ids = m_decode_ids.cpu()
                ref_logits, _, _ = self.ref_model(input_ids=m_input, decoder_input_ids=m_decode_ids)
            else:
                logits, _, v = self.model(m_input)
                ref_logits, _, _ = self.ref_model(m_input)

            m_input = m_input.to(self.device)
            ref_logits = ref_logits.to(self.device)

            print(f"{'-'*10}{'>'*10} move to {self.device}")

            values.append(v[:, -gen_len - 1:-1].detach())
            logprobs.append(logprobs_from_logits(logits[:, :-1, :], m_input[:, 1:])[:, -gen_len:].detach())
            ref_logprobs.append(logprobs_from_logits(ref_logits[:, :-1, :], m_input[:, 1:])[:, -gen_len:].detach())

        return torch.cat(logprobs), torch.cat(ref_logprobs), torch.cat(values)

    def train_minibatch(self, logprobs, values, rewards, query, response, model_input, decoder_input_ids=None):
        """Train one PPO minibatch"""
        loss_p, loss_v, train_stats = self.loss(logprobs, values, rewards, query, response, model_input,
                                                decoder_input_ids)
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_stats

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone().detach()
        rewards[:, -1] += scores
        return rewards, non_score_reward, self.kl_ctl.value

    def loss(self, old_logprobs, values, rewards, query, response, model_input, decoder_input_ids=None):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        if decoder_input_ids is not None:
            logits, _, vpred = self.model(input_ids=model_input, decoder_input_ids=decoder_input_ids)
        else:
            logits, _, vpred = self.model(model_input)
        logprob = logprobs_from_logits(logits[:, :-1, :], model_input[:, 1:])

        # only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:, -gen_len - 1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac = torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs) ** 2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward = torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats
