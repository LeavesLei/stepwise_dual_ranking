# stepwise_dual_ranking
[KDD 2026] Official repository for "[Offline Behavioral Data Selection](https://arxiv.org/pdf/2512.18246)" by Shiye Lei, Zhihao Cheng, and Dacheng Tao



## Dependencies

- Python 3.7
- Pytorch 1.11
- mujoco 2.10
- d4rl
- wandb



## Quick Start

- Near-expert policy $\pi^\ast$ checkpoints are provided in [offline_policy_checkpoints](./offline_policy_checkpoints) and obtained by using Cal-QL implemented in [CORL](https://github.com/tinkoff-ai/CORL).
- State density files are provided in [state_density](./state_density) and obtained by [Masked Autoregressive Flow](https://github.com/kamenbliznashki/normalizing_flows).



- **Stepwise Dual Ranking**

```shell
python select_bc.py --env 'halfcheetah-medium-replay-v2' --normalize True --policy_path_dir 'offline_policy_checkpoints/' --project 'SDR' --budget 1024 --lambda_1 0.2 --lambda_2 0.2 --training_epoch 20 --seed 0
```



- **Random Selection Baseline**

```shell
python select_bc_baseline.py --env 'halfcheetah-medium-replay-v2' --normalize True --policy_path_dir 'offline_policy_checkpoints/' --project 'SDR-Baseline' --budget 1024 --lambda_1 0 --lambda_2 0 --training_epoch 20 --seed 0
```



- **Test loss vs. Normalized Return (Figure 3)**

```shell
python subset_error_vs_return.py --env 'halfcheetah-medium-replay-v2' --normalize True --policy_path_dir 'offline_policy_checkpoints/' --project 'SDR' --subset_size 1024 --training_epoch 20 --seed 0
```



## Citation

```
@inproceedings{lei2026offline,
  title={Offline Behavioral Data Selection},
  author={Lei, Shiye and Cheng, Zhihao and Tao, Dacheng},
  booktitle={Proceedings of the 32th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```



## Contact

For any issue, please kindly contact Shiye Lei: [leishiye@gmail.com](mailto:leishiye@gmail.com)

