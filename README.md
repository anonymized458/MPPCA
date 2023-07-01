# Efficient Distribution Similarity Identification in Clustered Federated Learning via Principal Angles Between Client Data Subspaces (PACFL)

The official code of
paper ''[Neural Networks as Data: Identifying Malicious and Novel Clients in Heterogeneous Federated Learning]()''.

In this repository, we release the official implementation for MPPCA algorithm. We also release the implementation of
the following algorithms:

* FedAvg
* COS
* L2
* MI
* F-PCA

## Usage

We provide scripts to run the algorithms, which are put under `scripts/`. Here is an example to run the script:

```
cd scripts
bash fedavg.sh
```

Please follow the paper to modify the scripts for more experiments. You may change the parameters listed in the
following table.

The descriptions of parameters are as follows:
| Parameter | Description | | --------- | ----------- | | ntrials | The number of total runs. | | rounds | The number of
communication rounds per run. | | num_users | The number of clients. | | frac | The sampling rate of clients for each
round. | | local_ep | The number of local training epochs. | | local_bs | Local batch size. | | lr | The learning rate
for local models. | | momentum | The momentum for the optimizer. | | model | Network architecture. Options: simple-cnn,
resnet9 | | dataset | The dataset for training and testing. Options are discussed above. | | partition | How datasets
are partitioned. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of
labels each party owns). | | datadir | The path of datasets. | | logdir | The path to store logs. | | log_filename | The
folder name for multiple runs. E.g., with `ntrials=3` and `log_filename=$trial`, the logs of 3 runs will be located in 3
folders named `1`, `2`, and `3`. | | alg | Federated learning algorithm. Options are discussed above. | | beta | The
concentration parameter of the Dirichlet distribution for heterogeneous partition. | | local_view | If true puts local
test set for each client | | gpu | The IDs of GPU to use. E.g., 0 | | print_freq | The frequency to print training logs.
E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. | | times | The number of
federated learing running times for one script running ||

## Citation

Please cite our work if you find it relavent to your research and used our implementations.

## Acknowledgements

Some parts of our code and implementation has been adapted
from [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench) and [PACFL](https://github.com/MMorafah/PACFL)
repository.

## Contact 

