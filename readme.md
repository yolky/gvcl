# Generalized Variational Continual Learning

This is code for the paper Generalized Variational Continual Learning<sup>1</sup>. This repository is originally based on the [HAT<sup>2</sup> repository](https://github.com/joansj/hat).

[Link to paper](https://openreview.net/pdf?id=_IM-AfFhna9)

## Abstract

Continual learning deals with training models on new tasks and datasets in an online fashion. One strand of research has used probabilistic regularization for continual learning, with two of the main approaches in this vein being Online Elastic Weight Consolidation (Online EWC) and Variational Continual Learning (VCL). VCL employs variational inference, which in other settings has been improved empirically by applying likelihood-tempering. We show that applying this modification to VCL recovers Online EWC as a limiting case, allowing for interpolation between the two approaches. We term the general algorithm Generalized VCL (GVCL). In order to mitigate the observed overpruning effect of VI, we take inspiration from a common multi-task architecture, neural networks with task-specific FiLM layers, and find that this addition leads to significant performance gains, specifically for variational methods. In the small-data regime, GVCL strongly outperforms existing baselines. In larger datasets, GVCL with FiLM layers outperforms or is competitive with existing baselines in terms of accuracy, whilst also providing significantly better calibration.

## Authors

Noel Loo, Siddharth Swaroop, Richard E. Turner


## Installing

1. Create a python 3 conda environment (check the requirements.txt file)

2. To run chasy experiments, run src/dataloaders/hasy_utils.py to download the dataset

3. The following folder structure is expected at runtime. From the git folder:
    * src/ : Where all the scripts lie (already produced by the repo)
    * dat/ : Place to put/download all data sets
    * res/ : Place to save results
    * tmp/ : Place to store temporary files

4. The main script is src/run.py. To run multiple experiments we use src/run_multi.py or src/work.py; to run the compression experiment we use src/run_compression.sh.

## Notes

* The original HAT repository had mnist twice (instead of fashion mnist) for the mixed vision tasks so results on that benchmark may differ
* The the implementation of EWC and IMM-mode is different than the orginal repository, since since the original repository calculated the FIM using batches instead of individual samples
* The two ipython notebooks are for the toy examples in appendix A and B

## References

<sup>1</sup> Noel Loo, Siddharth Swaroop, & Richard E Turner (2021). Generalized Variational Continual Learning. In International Conference on Learning Representations.

<sup>2</sup> Serrà, J., Surís, D., Miron, M. & Karatzoglou, A.. (2018). Overcoming Catastrophic Forgetting with Hard Attention to the Task. Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:4548-4557