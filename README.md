# masters-creativity
Minimalistic Pytorch Lightning + Hydra + transformers template, for experiments with NLP, mainly text generation. 
Created for masters thesis about computional creativity.

#### Checkpoints note
When training, *.ckpt* files are checkpoint of the implemented model class - so e.g. CLMTransformer, they contain:
```"epoch", "global_step", "pytorch-lightning_version", "state_dict", "loops", "callbacks", "optimizer_states", "lr_schedulers", "NativeMixedPrecisionPlugin".```
*.pkl* saved at the end of a training, are checkpoints of the transformer model, so of *model* in class models