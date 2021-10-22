

## ProSelfLC: CVPR 2021


#### [ProSelfLC: Progressive Self Label Correction for Training Robust Deep Neural Networks](https://xinshaoamoswang.github.io/blogs/2020-06-07-Progressive-self-label-correction/)




For any specific discussion or potential future collaboration, please feel free to contact me. <br />

Paper link: [https://arxiv.org/abs/2005.03788](https://arxiv.org/abs/2005.03788)

#### Please kindly cite our work if you find the paper and this repo useful.
```
@inproceddings{wang2021proselflc,
  title={ {ProSelfLC}: Progressive Self Label Correction
  for Training Robust Deep Neural Networks},
  author={Wang, Xinshao and Hua, Yang and Kodirov, Elyor and Clifton, David A and Robertson, Neil M},
  booktitle={CVPR},
  year={2021}
}
```

# PyTorch Implementation for ProSelfLC-CVPR 2021
* Easy to install
* Easy to use
* Easy to extend: new losses, new networks, new dataset and loaders
* Easy to run experiments and sink results

## Install
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

## How to use
#### Run experiments
* cd ProSelfLC-CVPR2021
* pipenv shell
* `python tests/convnets/test_trainer_cnn_vision_cifar100_cce_convets.py`
* `python tests/convnets/test_trainer_cnn_vision_cifar100_lscplc_covnets.py`
* `python tests/convnets/test_trainer_cnn_vision_cifar100_proselflc_covnets.py`
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
* The results are well sinked and organised, e.g.,
[proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732)
* ![Accuracy curve: shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732/accuracy.pdf)
* ![Loss curve: shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732/loss.pdf)
* ![Accuracy and loss tables: shufflenetv2](proselflc_experiments/cifar100_symmetric_noise_rate_0.2/shufflenetv2/050_proselflc_warm16_b6_transit0.3_20210904-172732/accuracy_loss.html)



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
