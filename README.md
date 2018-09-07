# Pathologies of Neural Models Make Interpretations Difficult

This is the code for the 2018 EMNLP Paper, [Pathologies of Neural Models Make Interpretations Difficult](https://arxiv.org/abs/1804.07781). 

This repository contains the code for input reduction. If you want to apply input reduction to your task or model, we recommend
taking a look at the ```rawr.py``` file. From there, you can learn how to calculate gradients of the prediction with the respect to the input.

## Dependencies

This code is written in python using the highly underrated Chainer framework. If you know PyTorch, you will love it =).

Dependencies include:

* Python 2/3
* [Chainer](https://chainer.org/)
* numpy

A portion of the code is built off Chainers [text classification example](https://github.com/chainer/chainer/tree/master/examples/text_classification). See their documentation and code to understand the basic layout of our project. 

## References

Please consider citing [1] if you found this code or our work beneficial to your research.

### Pathologies of Neural Models Make Interpretations Difficult

[1] Shi Feng, Eric Wallace, Alvin Grissom II, Mohit Iyyer, Pedro Rodriguez, Jordan Boyd-Graber [Pathologies of Neural Models Make Interpretations Difficult](https://arxiv.org/abs/1804.07781). 

```
@article{feng2018pathologies,
  title={Pathologies of Neural Models Make Interpretations Difficult},
  author={Shi Feng and Eric Wallace and Alvin Grissom II and Mohit Iyyer and Pedro Rodriguez and Jordan Boyd-Graber},
  journal={Empirical Methods in Natural Language Processing},  
  year={2018},  
}
```

## Contact

For issues with code or suggested improvements, feel free to open a pull request.

To contact the authors, reach out to Shi Feng (shifeng@cs.umd.edu).
