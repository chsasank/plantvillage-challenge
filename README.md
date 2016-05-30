# Code in Torch for [PlantVillage challenge](https://www.crowdai.org/challenges/1)

I wrote a blog post describing the code here: http://chsasank.github.io/plantvillage-tutorial.html


## Requirements
See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [Torch](http://torch.ch/docs/getting-started.html)
- Optionally install Nvidia CUDA and [cuDNN v4](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
- Download and extract the training and test dataset from https://www.crowdai.org.

If you already have Torch installed, update `nn`, `cunn`, and `cudnn`.

Divide the training data in to `train` and `val` folders. You can use a bash script like this:
```bash
cd directory/contaning/c_0c_1...etcdirectories
mkdir -p train val
for i in {0..37}; do mkdir val/c_$i; done
mv c_* train
```

## Training
The training scripts come with several options, which can be listed with the `--help` flag.
```bash
$ th main.lua --help
```

```
Torch-7 PlantVillage Challenge Training script

  -learningRate initial learning rate for sgd [0.01]
  -momentum     momentum term of sgd [0.9]
  -maxEpochs    Max # Epochs [120]
  -batchSize    batch size [32]
  -nbClasses    # of classes [38]
  -nbChannels   # of channels [3]
  -backend      Options: cudnn | nn [cudnn]
  -model        Options: alexnet | vgg | resnet [alexnet]
  -depth        For vgg depth: A | B | C | D, For resnet depth: 18 | 34 | 50 | 101 | ... Not applicable for alexnet [A]
  -retrain      Path to model to finetune [none]
  -save         Path to save models [.]
  -data         Path to folder with train and val directories [datasets/crowdai/]

```

### Example usage
Train alexnet:
```bash
$ th main.lua -model alexnet -data path/to/train-val-directories
```

Train alexnet on CPU (not recommended):
```bash
$ th main.lua -model alexnet -data path/to/train-val-directories -backend nn
```

Train resnet 34
```bash
$ th main.lua -model resnet -depth 34 -learningRate 0.1 -data path/to/train-val-directories
```

This checkpoints the model every 10 epochs. It also saves best model as per validation set.

## Submission
Create a submission using `model.h5`:
```bash
th submission.lua model.h5 path/to/test > submission.csv
```
