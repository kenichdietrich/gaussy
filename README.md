# gaussy
## Gaussian Splatting made easy

### Usage

```python
from gaussy import read_scene, Gaussians, TrainConfig, train

scene = read_scene("./egypt")

gss = Gaussians().load_scene(scene).to("cuda")

train_config = TrainConfig()
logs = train(train_config, gss, scene)

gss.save_ply("./scene.ply")
```

### References

* Kerbl, Bernhard, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. "3d Gaussian Splatting for Real-Time Radiance Field Rendering." arXiv.org, August 8, 2023. https://arxiv.org/abs/2308.04079. 