<div align="center">

# Model Explainability

</div>

In this repository, we perform model explainability for Image and NLP tasks.


## Image

```
pip install timm shap grad-cam captum
```

Images were downloaded from [github](https://github.com/EliSchwartz/imagenet-sample-images)

```
python explain.py
```

To see the logbook, open [explanation.md](./explanation.md)

## NLP

```
pip install transformers shap sentencepiece datasets
```

See the notebook `NLP.ipynb`
