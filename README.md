
## Deep Robustness: ProSelfLC-CVPR 2021 + Example Weighting (DM+IMAE)

#### [ProSelfLC: Progressive Self Label Correction for Training Robust Deep Neural Networks](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/)
* Paper link: [https://arxiv.org/abs/2005.03788](https://arxiv.org/abs/2005.03788)
#### [Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters](https://github.com/XinshaoAmosWang/Improving-Mean-Absolute-Error-against-CCE)
#### [Derivative Manipulation: Example Weighting via Emphasis Density Funtion in the context of DL](https://github.com/XinshaoAmosWang/DerivativeManipulation)
* Novelty: moving from loss design to derivative design




For any specific discussion or potential future collaboration, please feel free to contact me. <br />

<details><summary>See Citation Details</summary>

#### Please kindly cite the following papers if you find this repo useful.
```
@inproceddings{wang2021proselflc,
  title={ {ProSelfLC}: Progressive Self Label Correction
  for Training Robust Deep Neural Networks},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Clifton, David A and Robertson, Neil M},
  booktitle={CVPR},
  year={2021}
}
@phdthesis{wang2020example,
  title={Example weighting for deep representation learning},
  author={Wang, Xinshao},
  year={2020},
  school={Queen's University Belfast}
}
@article{wang2019derivative,
  title={Derivative Manipulation for General Example Weighting},
  author={Wang, Xinshao and Kodirov, Elyor and Hua, Yang and Robertson, Neil},
  journal={arXiv preprint arXiv:1905.11233},
  year={2019}
}
@article{wang2019imae,
  title={{IMAE} for Noise-Robust Learning: Mean Absolute Error Does Not Treat Examples Equally and Gradient Magnitude’s Variance Matters},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Robertson, Neil M},
  journal={arXiv preprint arXiv:1903.12141},
  year={2019}
}
```
</details>

## PyTorch Implementation for ProSelfLC, Derivative Manipulation, Improved MAE
* Easy to install
* Easy to use
* Easy to extend: new losses, new networks, new dataset and loaders
* Easy to run experiments and sink results
* Easy to put sinked results into your technical reports and academic papers.

## Install

<details><summary>See Install Guidelines</summary>

#### Set the Pipenv From Scratch
* sudo apt update && sudo apt upgrade
* sudo apt install python3.8
* curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
* python3.8 get-pip.py
* vim ~/.bashrc -> add `export PATH="/home/ubuntu/.local/bin:$PATH"` -> source ~/.bashrc
* pip3 install pipenv

#### Build env for this repo using pipenv
* git clone `git@github.com:XinshaoAmosWang/ProSelfLC-CVPR2021.git`
* cd ProSelfLC-CVPR2021
* pipenv install -e .

</details>

## How to use
#### Run experiments
* cd ProSelfLC-CVPR2021
* pipenv shell
* `python tests/convnets/test_trainer_cnn_vision_cifar100_cce_convets.py`
* `python tests/convnets/test_trainer_cnn_vision_cifar100_lscplc_covnets.py`
* `python tests/convnets/test_trainer_cnn_vision_cifar100_proselflc_covnets.py`
* `python tests/derivativemanipulation_imae/test_dm_imae.py`
#### Visualize results
* The results are well sinked and organised, e.g.,
[proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732)


## How to extend this repo
* Add dataset and dataloader: see examples in [src/proselflc/slices/datain](src/proselflc/slices/datain)
* Add losses: see examples in [src/proselflc/slices/losses](src/proselflc/slices/losses)
* Add networks: see examples in [src/proselflc/slices/networks](src/proselflc/slices/networks)
* Add optimisers: see examples in [src/proselflc/optim](src/proselflc/optim)
* Extend the slicegetter: [src/proselflc/slicegetter](src/proselflc/slicegetter)
* Write run scripts: see examples in [tests/convnets/](tests/convnets/)



## Examples of sinked experimental configs and resutls
<details><summary>See Sinked Results</summary>

* The results are well sinked and organised, e.g.,
[proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732)

* ![Accuracy curve: shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732/accuracy.pdf)


* ![Loss curve: shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732/loss.pdf)

* accuracy_loss.xlsx

| epoch | clean_test | noisy_subset | clean_subset | cleaned_noisy_subset |
|-------|------------|--------------|--------------|----------------------|
| 4     | 0.1801     | 0.0091       | 0.169        | 0.1617               |
| 8     | 0.3329     | 0.0089       | 0.32655      | 0.3065               |
| 12    | 0.3848     | 0.0093       | 0.380025     | 0.3451               |
| 16    | 0.391      | 0.0091       | 0.391875     | 0.3561               |
| 20    | 0.4225     | 0.009        | 0.4119       | 0.3693               |
| 24    | 0.4214     | 0.0077       | 0.416975     | 0.3728               |
| 28    | 0.4222     | 0.0108       | 0.43095      | 0.3884               |
| 32    | 0.4709     | 0.0097       | 0.47         | 0.4254               |
| 36    | 0.4155     | 0.0097       | 0.42955      | 0.3886               |
| 40    | 0.449      | 0.0083       | 0.463125     | 0.4165               |
| 44    | 0.448      | 0.0074       | 0.441125     | 0.4006               |
| 48    | 0.3856     | 0.0078       | 0.383025     | 0.3496               |
| 52    | 0.4672     | 0.0083       | 0.479475     | 0.4373               |
| 56    | 0.4428     | 0.0081       | 0.437075     | 0.3891               |
| 60    | 0.4164     | 0.0095       | 0.422675     | 0.3815               |
| 64    | 0.4635     | 0.0079       | 0.483225     | 0.4386               |
| 68    | 0.4506     | 0.0085       | 0.4654       | 0.4145               |
| 72    | 0.4428     | 0.0081       | 0.459825     | 0.4105               |
| 76    | 0.4553     | 0.0086       | 0.4579       | 0.4151               |
| 80    | 0.6108     | 0.0104       | 0.670775     | 0.5751               |
| 84    | 0.5989     | 0.0107       | 0.67395      | 0.5636               |
| 88    | 0.6026     | 0.0098       | 0.6832       | 0.5703               |
| 92    | 0.5949     | 0.013        | 0.679925     | 0.5608               |
| 96    | 0.5976     | 0.0122       | 0.67985      | 0.5487               |
| 100   | 0.5838     | 0.0123       | 0.67         | 0.5426               |
| 104   | 0.592      | 0.0113       | 0.67465      | 0.5544               |
| 108   | 0.603      | 0.0117       | 0.6723       | 0.5552               |
| 112   | 0.58       | 0.0117       | 0.662175     | 0.5394               |
| 116   | 0.5767     | 0.0126       | 0.660825     | 0.5421               |
| 120   | 0.5829     | 0.0121       | 0.654925     | 0.5382               |
| 124   | 0.5828     | 0.0127       | 0.655875     | 0.5426               |
| 128   | 0.5825     | 0.013        | 0.652575     | 0.5405               |
| 132   | 0.5641     | 0.0111       | 0.625275     | 0.5303               |
| 136   | 0.5779     | 0.0112       | 0.635275     | 0.5355               |
| 140   | 0.6462     | 0.0139       | 0.76175      | 0.6095               |
| 144   | 0.6464     | 0.0159       | 0.76965      | 0.6175               |
| 148   | 0.6412     | 0.0169       | 0.773475     | 0.616                |
| 152   | 0.6458     | 0.0161       | 0.775025     | 0.6127               |
| 156   | 0.6353     | 0.019        | 0.768825     | 0.6036               |
| 160   | 0.6385     | 0.0168       | 0.768125     | 0.6115               |
| 164   | 0.6338     | 0.0181       | 0.763825     | 0.6122               |
| 168   | 0.6316     | 0.0164       | 0.75755      | 0.6011               |
| 172   | 0.6225     | 0.0171       | 0.747675     | 0.5942               |
| 176   | 0.6312     | 0.0153       | 0.749425     | 0.5989               |
| 180   | 0.6552     | 0.021        | 0.7966       | 0.6264               |
| 184   | 0.653      | 0.022        | 0.8036       | 0.6292               |
| 188   | 0.6544     | 0.0212       | 0.807125     | 0.6256               |
| 192   | 0.6545     | 0.0209       | 0.80905      | 0.6286               |
| 196   | 0.6531     | 0.0222       | 0.811075     | 0.6276               |
| 200   | 0.6572     | 0.0229       | 0.8102       | 0.627                |

* params.csv


| data_name | num_classes | device | num_workers | batch_size | counter   | lr  | total_epochs | milestones | gamma | loss_name | symmetric_noise_rate | network_name | warmup_epochs | exp_base | transit_time_ratio | summary_writer_dir                                                                                                                          | train | total_iterations | momentum | weight_decay |
|-----------|-------------|--------|-------------|------------|-----------|-----|--------------|------------|-------|-----------|----------------------|--------------|---------------|----------|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------|------------------|----------|--------------|
| cifar100  | 100         | gpu    | 4           | 128        | iteration | 0.1 | 200          | 60         | 0.2   | proselflc | 0.2                  | shufflenetv2 | 16            | 6        | 0.3                | /home/xinshao/tpami_proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732 | True  | 78200            | 0.9      | 0.0005       |
| cifar100  | 100         | gpu    | 4           | 128        | iteration | 0.1 | 200          | 120        | 0.2   | proselflc | 0.2                  | shufflenetv2 | 16            | 6        | 0.3                | /home/xinshao/tpami_proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732 | True  | 78200            | 0.9      | 0.0005       |
| cifar100  | 100         | gpu    | 4           | 128        | iteration | 0.1 | 200          | 160        | 0.2   | proselflc | 0.2                  | shufflenetv2 | 16            | 6        | 0.3                | /home/xinshao/tpami_proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732 | True  | 78200            | 0.9      | 0.0005       |


</details>


#### [Link to Slide, Poster, Final version](./Poster_Slide)

#### [Link to reviewers' comments](./Reviews)

#### List of Content

<!-- :+1: means being highly related to my personal research interest. -->
0. [Storyline](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#storyline)
0. [Open ML Research Questions](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#open-ml-research-questions)
0. [Noticeable Findings](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#noticeable-findings)
0. [Literature Review](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#literature-review)
0. [In Self LC, a core question is not well answered](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#in-self-lc-a-core-question-is-not-well-answered)
0. [Underlying Principle of ProSelfLC](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#underlying-principle-of-proselflc)
0. [Mathematical Details of ProSelfLC](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#mathematical-details-of-proselflc)
0. [Design Reasons of ProSelfLC](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#design-reasons-of-proselflc)
0. [Related Interesting Work](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/#related-interesting-work)
