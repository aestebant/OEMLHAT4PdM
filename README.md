# OEMLHAT4PdM: Online Ensemble of Multi-Label Hoeffding Trees for Predictive Maintenance

Associated repository with complementary material to the manuscript *Simultaneous fault prediction in evolving industrial environments with ensembles of Hoeffding adaptive trees*, submitted to the Applied Intelligence journal. The following materials are included:

* Source code of the OEMLHAT proposal.
* Datasets used in the experimentation.
* Complete table of results.
* Complete instructions to execute the model and reproduce the experimentation.

## Source code

The purpose of this repository is to make public and accesible the source code of OEMLHAT. This includes the dependencies of the library and the necessary instructions to use it.

The source code of OEMLHAT is available under the file [src/oemlhat](src/oemlhat.py). And a complete tutorial for its execution is presented in the [Tutorial notebook](src/tutorial.ipynb)

```python
from oemlhat import OEMLHAT

learner = OEMLHAT(
    n_models = 10,
    subspace_size = 0.6,
    lam = 5,
    grace_period = 205,
    delta = 2.5093683326920607e-07,
    tau = 0.05,
    cardinality_th = 135,
    entropy_th = 0.75,
    drift_window_threshold = 208,
    perf_metric = MicroAverage(F1()),
    switch_significance = 0.036,
    lc_n_neighbors = 13,
    lc_window_size = 140,
    hc_lr = 0.95,
)
```

## Datasets

OEMLHAT's performance has been tested on three public predictive maintenance problems belonging to the machine learning field of the multi-label classification. All of them are available in this repository under the [datasets](datasets) folder. The dataset information from a point of view of multi-label learning is the following:

| Case study | Reference | Stream length | Numerical features | Categorical features | Labels | Cardinality | Density |
|-----------|-----|-----------------|----------------------|------------------------|--------|-------------|---------|
| Ai4i      | [UCI repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) | 10,000          | 5                    | 1                      | 5      | 3.75e-2     | 7.46e-3 |
| NPS     | [UCI repository](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants) | 65,473          | 25                   | 0                      | 4      | 2.00        | 0.50    |
| ALPI   | [IEEE dataport](https://ieee-dataport.org/open-access/alarm-logs-packaging-industry-alpi) | 3,395           | 36                   | 0                      | 47     | 2.83        | 6.16e-2 |

Moreover, they can be loaded as data stream in River framework using the functions under the folder [src/datasets](src/datasets/)

```python
from datasets.multioutput import Ai4i

stream = Ai4i()

x, y = next(iter(stream))
```

## Results

The complete results of the experimentation carried out in this work and presented and discussed in the associated paper are available in CSV format for downloaading in the [results](results) folder. These are the average results over the tree tested problems.

| Model           | Subset acc      | Hamming loss    | Example acc     | Example F1      | Micro F1        | Micro precision | Micro recall  | Macro F1        | Macro precision | Macro recall    |
|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|---------------|-----------------|-----------------|-----------------|
| HT              | 0.6698          | 0.0365          | 0.9586          | 0.7935          | 0.5138          | 0.6617          | 0.4670        | 0.4124          | 0.6040          | 0.4067          |
| HAT             | 0.6322          | 0.0472          | 0.9478          | 0.7778          | 0.5033          | 0.6549          | 0.4526        | 0.4030          | 0.5978          | 0.3939          |
| AMR             | 0.6947          | 0.0339          | 0.9611          | 0.8429          | 0.6433          | 0.6554          | **0.6318** | **0.5894** | 0.5978          | **0.5857** |
| KNN             | 0.6368          | 0.0504          | 0.9447          | 0.7875          | 0.5778          | 0.7061          | 0.5317        | 0.4490          | 0.6242          | 0.4354          |
| MLHT            | 0.5022          | 0.1213          | 0.8737          | 0.6536          | 0.2450          | 0.8457          | 0.1758        | 0.1912          | **0.9603** | 0.1515          |
| iSOUPT          | 0.7101          | 0.0295          | 0.9656          | 0.8397          | 0.6248          | 0.6249          | 0.6248        | 0.5690          | 0.5655          | 0.5752          |
| $OEMLHAT_{BA}$  | 0.6953          | 0.0342          | 0.9609          | 0.8292          | 0.5916          | 0.8090          | 0.5323        | 0.4737          | 0.8403          | 0.4459          |
| $OEMLHAT_{SRP}$ | **0.7549** | **0.0219** | **0.9732** | **0.8736** | 0.6285          | **0.8517** | 0.5888        | 0.5752          | 0.8510          | 0.5527          |
| $OEMLHAT_{BO}$  | 0.7149          | 0.0281          | 0.9670          | 0.8394          | **0.6799** | 0.7773          | 0.6237        | 0.5602          | 0.6726          | 0.5198          |

## Reproductible experimentation

All the experimentation has been carried out in Python, using for the comparative analysis in the online learning paradigm the implementations available in [River](https://riverml.xyz) of the main multi-label methods. The methods together with their configurations are the following:

| Family      | Algorithm  | Configuration |
|-------------|------------|--------|
| BR+DT       | [Hoeffding Tree](https://riverml.xyz/latest/api/tree/HoeffdingTreeClassifier/)   | `grace_period` = 200, `delta` = 1e-7, `split_criterion` =  'info gain', `tau` = 0.05, `leaf_prediction` = 'nba'|
| BR+DT       | [Hoeffding Adaptive Tree](https://riverml.xyz/latest/api/tree/HoeffdingAdaptiveTreeClassifier/)  | `grace_period`= 200, `delta`= 1e-7, `split_criterion` = 'info gain', `tau`= 0.05, `leaf_prediction` = 'nba', `bootstrap_sampling` = True, `drift_window_threshold` = 300, `switch_significance` = 0.05 |
| BR+Distance | [k-Nearest Neighbors](https://riverml.xyz/latest/api/neighbors/KNNClassifier/)  | `n_neighbors` = 5, `window_size` = 200, `weighted` = True, `dist_func` = Euclidean |
| BR+Rules    | [Adaptive Model Rules](https://riverml.xyz/latest/api/rules/AMRules/)  | `n_min` = 200, `delta` = 1e-7, `tau` = 0.05, `pred_model` = logistic regression |
| AA+DT       | [Multi-Label Hoeffding Tree](src/multioutput/mlht.py) | `grace_period` = 200, `delta` = 1e-5, `leaf_model` = pruned set of Hoeffding tree (`grace_period` = 100, `delta`= 1e-5) |
| AA+DT        | [iSOUPT](https://riverml.xyz/latest/api/tree/iSOUPTreeRegressor/)        | `grace_period` = 200, `delta` = 1e-5, `leaf_model` = logistic regression |
