import matplotlib.pyplot as plt
from torchinfo import summary
from torchview import draw_graph
import torch

from src.models.dummy import DummyModel
from src.models.resnet import ResnetLikeV1


IMAGE_SHAPE, N_CLASS = (224, 224), 2

model = DummyModel(N_CLASS, IMAGE_SHAPE)

x = torch.randn(1, 3, *IMAGE_SHAPE)
summary(model, input_data=x)
graph = draw_graph(model, x)
fig = graph.visual_graph
plt.show()
