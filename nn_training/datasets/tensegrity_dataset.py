import torch
import tqdm
from torch.utils.data import Dataset

from utilities.misc_utils import DEFAULT_DTYPE, compute_num_steps


class TensegrityDataset(Dataset):

    def __init__(self,
                 raw_data_dict,
                 num_steps_fwd=1,
                 dt=0.01,
                 num_ctrls_hist=20,
                 dtype=DEFAULT_DTYPE):
        super().__init__()
        self.num_steps_fwd = num_steps_fwd
        self.dt = dt
        self.dtype = dtype
        self.num_ctrls_hist = num_ctrls_hist

        self.processed_data = self._process_raw_data(raw_data_dict)

    def _process_raw_data(self, raw_data_dict):
        processed_data = []

        times = raw_data_dict['times']
        data_gt_end_pts = raw_data_dict["gt_end_pts"]

        for i in tqdm.tqdm(range(len(raw_data_dict['states'])), desc='Loading torch dataset'):
            dataset_idx = torch.tensor([raw_data_dict['dataset_idx'][i]])
            controls = torch.vstack(raw_data_dict['controls'][i])

            end = controls.shape[0] if self.num_steps_fwd == 1 else -self.num_steps_fwd + 1
            all_ctrls_hist = raw_data_dict['controls'][i][:end]
            all_ctrls_hist = [torch.zeros_like(all_ctrls_hist[0])
                              for _ in range(self.num_ctrls_hist)
                              ] + all_ctrls_hist

            act_lengths = raw_data_dict['act_lengths'][i][:-self.num_steps_fwd]
            x = raw_data_dict['states'][i][:-self.num_steps_fwd]

            gt_end_pts = torch.vstack([torch.hstack(d) for d in data_gt_end_pts[i]])
            next_act_lengths = torch.vstack(raw_data_dict['act_lengths'][i])

            traj_x = [torch.concat(x[j: j + 1], 2) for j in range(len(x))]
            traj_act_lens = [torch.concat(act_lengths[j: j + 1], 2)
                             for j in range(len(act_lengths))]
            traj_ctrls_hist = [torch.concat(all_ctrls_hist[j - self.num_ctrls_hist: j], 2)
                               if self.num_ctrls_hist > 0 else torch.empty(all_ctrls_hist[0].shape)
                               for j in range(self.num_ctrls_hist, len(all_ctrls_hist))]

            traj_y, traj_ctrls = [], []
            traj_next_act_lens = []
            for j in range(1, self.num_steps_fwd + 1):
                end = -(self.num_steps_fwd - j) \
                    if j < self.num_steps_fwd else gt_end_pts.shape[0]
                traj_y.append(gt_end_pts[j:end])
                traj_next_act_lens.append(next_act_lengths[j:end])
                traj_ctrls.append(controls[j - 1: end])

            traj_y = torch.concat(traj_y, dim=2)
            traj_next_act_lens = torch.concat(traj_next_act_lens, dim=2)
            traj_ctrls = torch.concat(traj_ctrls, dim=2)

            traj_y = [traj_y[j: j + 1] for j in range(traj_y.shape[0])]
            traj_next_act_lens = [traj_next_act_lens[j: j + 1]
                                  for j in range(traj_next_act_lens.shape[0])]
            traj_ctrls = [traj_ctrls[j: j + 1] for j in range(traj_ctrls.shape[0])]

            traj_dt = [times[i][j + self.num_steps_fwd] - times[i][j]
                       for j in range(len(times[i]) - self.num_steps_fwd)]

            traj_elms = zip(traj_x, traj_y, traj_dt, traj_ctrls,
                            traj_act_lens, traj_next_act_lens,
                            traj_ctrls_hist)
            for x, y, dt, ctrl, act_len, next_act_lens, ctrls_hist in traj_elms:
                delta_t = compute_num_steps(dt, self.dt) * self.dt

                data_instance = {
                    'x': x,
                    'y': y,
                    'ctrl': ctrl,
                    'dt': torch.tensor([[[dt]]], dtype=self.dtype),
                    'act_len': act_len,
                    'next_act_lens': next_act_lens,
                    'delta_t': torch.tensor([[[delta_t]]], dtype=self.dtype),
                    'ctrls_hist': ctrls_hist,
                    'dataset_idx': dataset_idx,
                }
                processed_data.append(data_instance)

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index):
        return self.processed_data[index]

    @staticmethod
    def collate_fn(data_instances):
        batch = {
            k: torch.vstack([d[k] for d in data_instances])
            for k in data_instances[0].keys()
        }

        return batch

