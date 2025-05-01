# SClassifier, reset_weightsimport torch
from torch import nn
import timm
import torch.nn as nn




class ESCModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3, activation_fn=nn.ReLU, use_residual=True):
        super(ESCModel, self).__init__()
        self.use_residual = use_residual
        self.fc_layers = self._create_fc_layers(input_size, hidden_sizes, dropout_prob, activation_fn)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation_fn = activation_fn()
        self.softmax = nn.LogSoftmax(dim=1)

    def _create_fc_layers(self, input_size, hidden_sizes, dropout_prob, activation_fn):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  
            layers.append(activation_fn())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.fc_layers(x)
        if self.use_residual and residual.shape == x.shape:
            x += residual 
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def count_parameters(model, message=""):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{message} Trainable params: {trainable_params} of {total_params}")
    
class QClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3, activation_fn=nn.ReLU, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        # Store quantization parameters
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int32))
        
        # Network architecture
        self.fc_layers = self._create_fc_layers(input_size, hidden_sizes, dropout_prob, activation_fn)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation_fn = activation_fn()
        self.softmax = nn.Softmax(dim=1)

    def _create_fc_layers(self, input_size, hidden_sizes, dropout_prob, activation_fn):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                activation_fn(),
                nn.Dropout(p=dropout_prob)
            ])
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        # Single dequantization at network entry
        if x.is_quantized:
            x = x.dequantize()
        
        residual = x
        x = self.fc_layers(x)
        
        if self.use_residual and residual.shape == x.shape:
            x += residual
            
        x = self.output_layer(x)
        return self.softmax(x)
    
class SClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3, activation_fn=nn.ReLU, use_residual=True):
        super(SClassifier, self).__init__()
        self.use_residual = use_residual
        self.fc_layers = self._create_fc_layers(input_size, hidden_sizes, dropout_prob, activation_fn)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation_fn = activation_fn()
        self.softmax = nn.Softmax(dim=1)

    def _create_fc_layers(self, input_size, hidden_sizes, dropout_prob, activation_fn):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  
            layers.append(activation_fn())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.fc_layers(x)
        if self.use_residual and residual.shape == x.shape:
            x += residual 
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
    
class v1Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3):
        super(v1Classifier, self).__init__()
        self.fc_layers = self._create_fc_layers(input_size, hidden_sizes, dropout_prob)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.softmax = nn.Softmax(dim=1)

    def _create_fc_layers(self, input_size, hidden_sizes, dropout_prob):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))  # Dropout
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
    
def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_units, num_classes, dropout_rate):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, hidden_units)  # Input dimension is the embedding size (1536)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax across the class dimension
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        outputs = self.softmax(x)
        return outputs
def load_and_initialize_model(model_name, weights_path, feat_space):
    model = timm.create_model('mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', pretrained=False, num_classes=0)

    # Count parameters before loading the checkpoint
    count_parameters(model, message="Before loading checkpoint")

    checkpoint = torch.load(weights_path, map_location='cpu')
    checkpoint_model = checkpoint['model']

    # Count parameters after loading the checkpoint
    count_parameters(model, message="After loading checkpoint")

    # Initialize the extract_embed with the base model and new classifier
    model = classifier_embeddings(model, feat_space, model_name)
    # Load updated checkpoint into the model
    model.load_state_dict(checkpoint_model, strict=False)

    # Count parameters of the custom model
    count_parameters(model, message="Custom model parameters")
    
    return model

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def initialize_model(model_name, feat_space, MODEL_CONSTRUCTORS):
    if model_name in MODEL_CONSTRUCTORS:
        model_constructor = MODEL_CONSTRUCTORS[model_name]
        if model_name == "vit_h_14":
            from torchvision.models import vit_h_14, ViT_H_14_Weights
            weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.DEFAULT
            model = vit_h_14(weights=weights)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            preprocess = weights.transforms()
            data_config = None
            transforms = None
        elif model_name == "regnet_y_128gf":
            from torchvision.models import regnet_y_128gf, RegNet_Y_128GF_Weights
            weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
            model = regnet_y_128gf(weights=weights)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            preprocess = weights.transforms()
            data_config = None
            transforms = None
        elif model_name == "mobilenet_v3_large":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            model = mobilenet_v3_large(weights=weights)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            preprocess = weights.transforms()
            data_config = None
            transforms = None
        elif model_name in ("mobilenetv4_r448", "eva02_large_patch14_448_embeddings_imageNet"):
            model = model_constructor
            preprocess=None
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            model = classifier_embeddings(base_model=model, feat_space=feat_space, model_name=model_name)
        elif model_name == "mobilenetv4_r448_trained":
            # Pre-trained model with 5 classes
            weights_path = '/home/sebastian/codes/QuantumVE/q_Net/pretrain/mobilenetv4_r448/checkpoint-99.pth'
            model = load_and_initialize_model(model_name, weights_path, 5)
            
            preprocess=None
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            model = classifier_embeddings(base_model=model.base_model, feat_space=feat_space, model_name=model_name)
        else:
            model = model_constructor(pretrained=True, progress=True)
            model.classifier[1].in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_features=feat_space)
            preprocess = None
            data_config = None
            transforms = None
        return model, preprocess, transforms, data_config
    else:
        print("Model not available")
        return None
