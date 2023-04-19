import tensorrt as trt
import onnx
import torch


onnx_model = ""

device = torch.device('cuda:0')

onnx_model = onnx.load(onnx_model)

# 创建tensorrt构建器
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# 解析onnx
parser = trt.OnnxParser(network, logger)
if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

# TensorRT的配置器 --> 最大工作空间、最大batch、精度模式
config = builder.create_builder_config()
config.max_workspace_size = 1 << 20
config.max_batch_size = 1

# TensorRT网络优化配置 --> 接受不同shape的input、动态epoch
profile = builder.create_optimization_profile()
profile.set_shape('input', (1, 3, 224, 224), (4, 3, 224, 224))
config.add_optimazation(profile)

with torch.cuda.device(device):
    engine = builder.build_engine(network, config)

with open("model.engine", "wb") as f:
    f.write(bytearray(engine.serialize()))
    print('Finished!')



parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
with open("model.onnx", "rb") as model:
    parser.parse(model.read())
engine = builder.build_cuda_engine(network)