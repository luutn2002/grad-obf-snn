from spikingjelly.activation_based import functional, neuron, functional, surrogate, layer
from torch import nn
import spikingjelly.activation_based.model.spiking_resnet
import source.sew_resnet

class DefaultAtanIFNode(neuron.IFNode):
  def __init__(self):
        super().__init__(surrogate_function=surrogate.ATan())

class GeneralUndefendedSNNModel(nn.Module):
    def __init__(self,
                 num_class: int,
                 variant: str = "sew",
                 pool: str = "max",
                 layer_count: int = 18):
        super().__init__()
        match variant:
            case "sew":
                match pool:
                    case "max": pool_fn = layer.MaxPool2d
                    case "avg": pool_fn = layer.AvgPool2d
                    case default: Exception("Unrecognized string, check config again.")

                self.snn_fn = getattr(source.sew_resnet, f'sew_resnet{layer_count}')
                self.snn = self.snn_fn(pretrained=False,
                                        cnf="ADD",
                                        spiking_neuron=neuron.IFNode, 
                                        num_classes=num_class,
                                        surrogate_function=surrogate.ATan(),
                                        pooling_layer=pool_fn, 
                                        detach_reset=True)
            case "spk":
                self.snn_fn = getattr(spikingjelly.activation_based.model.spiking_resnet, f'spiking_resnet{layer_count}')
                self.snn = self.snn_fn(pretrained=False,
                                        spiking_neuron=DefaultAtanIFNode, 
                                        num_classes=num_class)
            case default:
                raise Exception("Unrecognized string, check config again.")
        functional.set_step_mode(self.snn, step_mode='m')
        
    def forward(self, x):
        return self.snn(x).mean(0)