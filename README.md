# Deep Imitation Learning for Humanoid Loco-manipulation through Human Teleoperation
[Mingyo Seo](https://mingyoseo.com), [Steve Han](https://www.linkedin.com/in/stevehan2001), [Kyutae Sim](https://www.linkedin.com/in/kyutae-sim-888593166), [Seung Hyeon Bang](https://sites.utexas.edu/hcrl/people/), [Carlos Gonzalez](https://sites.utexas.edu/hcrl/people/), [Luis Sentis](https://sites.google.com/view/lsentis), [Yuke Zhu](https://www.cs.utexas.edu/~yukez)

[Project](https://ut-austin-rpl.github.io/TRILL) | [arXiv](https://arxiv.org/abs/2309.01952)

![intro](docs/imgs/approach.png)

## Abstract
We tackle the problem of developing humanoid loco-manipulation skills with deep imitation learning. The challenge of collecting human demonstrations for humanoids, in conjunction with the difficulty of policy training under a high degree of freedom, presents substantial challenges. We introduce TRILL, a data-efficient framework for learning humanoid loco-manipulation policies from human demonstrations. In this framework, we collect human demonstration data through an intuitive Virtual Reality (VR) interface. We employ the whole-body control formulation to transform task-space commands from human operators into the robot's joint-torque actuation while stabilizing its dynamics. By employing high-level action abstractions tailored for humanoid robots, our method can efficiently learn complex loco-manipulation skills. We demonstrate the effectiveness of TRILL in simulation and on a real-world robot for performing various types of tasks. 

If you find our work useful in your research, please consider [citing](#citing).


## Dependencies
- Python 3.8.5 (recommended)
- [Robosuite  1.4.0](https://github.com/ARISE-Initiative/robosuite/tree/v1.4)
- [Robomimic 0.2.0](https://github.com/ARISE-Initiative/robomimic/tree/v0.2.0)
- [PyTorch](https://github.com/pytorch/pytorch)


## Usage

Please see [Getting Started](docs/Getting-Started.md).


## Dataset and Pre-trained Models
We provide our demonstration dataset in the `door` simulation environment ([link](https://utexas.box.com/s/3610huk9fu33m6wic16oe7crx8cahpl8)) and trained models of the Visuomotor Policies ([link](https://utexas.box.com/s/qn3156sxpejx4zf4piq5zh97srl5zcto)). We also plan to open our demonstration dataset and trained models in the `workbench` simulation environment in the near future.


## Implementation Details
Please see [this page](docs/Implementation-Details.md) for detailed information on our implementation, including the whole-body controller, model architecture, and teleoperation system.


## Related Repository
The implementation of the whole-body control is based on [PyPnC](https://github.com/junhyeokahn/PyPnC).


## Citing
```
@inproceedings{seo2023trill,
   title={Deep Imitation Learning for Humanoid Loco-manipulation through Human Teleoperation},
   author={Seo, Mingyo and Han, Steve and Sim, Kyutae and 
           Bang, Seung Hyeon and Gonzalez, Carlos and 
           Sentis, Luis and Zhu, Yuke},
   booktitle={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
   year={2023}
}
```
