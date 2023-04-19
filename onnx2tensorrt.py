import torch
import onnx
import tensorrt as trt
import argparse
import numpy as np


def normal_engine():
    onnx_model = 'model_data/models.onnx'

    class NaiveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(2, 2)

        def forward(self, x):
            return self.pool(x)

    device = torch.device('cuda:0')

    # generate ONNX model
    # torch.onnx.export(NaiveModel(), torch.randn(1, 3, 224, 224), onnx_model, input_names=['input'], output_names=['output'],
    #                   opset_version=11)
    onnx_model = onnx.load(onnx_model)

    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    # TensorRT配置器 --> 最大工作空间、最大batch、精度模式
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20

    # TensorRT网络优化配置 --> 接受不同的input shape、动态batch
    profile = builder.create_optimization_profile()
    profile.set_shape('input', [1, 3, 224, 224], [1, 3, 224, 224], [1, 3, 224, 224])
    config.add_optimization_profile(profile)

    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('model_data/model.engine', mode='wb') as f:    # engine为TensorRT序列化后的文件
        f.write(bytearray(engine.serialize()))
        print("generating file done!")


def fp16():
    # 解析输入参数
    parser = argparse.ArgumentParser(description='将FP32模型转换为FP16 TensorRT引擎文件')
    parser.add_argument('--model_file', type=str, required=True,
                        help='FP32模型文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出TensorRT引擎文件路径')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小')
    parser.add_argument('--input_shape', type=str, default='3,224,224',
                        help='输入张量形状')
    parser.add_argument('--output_shape', type=str, default='1000',
                        help='输出张量形状')
    args = parser.parse_args()
    # 加载FP32模型
    import torch
    model = torch.load(args.model_file)

    # 创建TensorRT构建器
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))

    # 设置构建器选项
    builder.max_batch_size = args.batch_size
    builder.max_workspace_size = 1 << 30

    # 创建TensorRT网络
    network = builder.create_network()

    # 创建输入张量
    input_shape = tuple(map(int, args.input_shape.split(',')))
    input = network.add_input(name='input', dtype=trt.float32, shape=input_shape)

    # 添加FP32模型到网络中
    output = model(input)

    # 创建输出张量
    output_shape = tuple(map(int, args.output_shape.split(',')))
    output = network.add_output(name='output', dtype=trt.float32, shape=output_shape)

    # 将网络转换为FP16精度
    builder.fp16_mode = True
    builder.strict_type_constraints = True

    # 构建TensorRT引擎
    engine = builder.build_cuda_engine(network)
    engine = builder.build_engine(network, config=None)

    # 保存TensorRT引擎文件
    with open(args.output_file, 'wb') as f:
        f.write(engine.serialize())


def fp16_2():
    # Load the TensorRT engine
    with trt.Builder(trt.Logger(trt.Logger.WARNING)) as builder, \
            builder.create_network() as network, \
            trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING)) as parser, \
            builder.create_builder_config() as builder_config:
        # builder_config.max_workspace_size = 1 * (1024 * 1024)
        # builder_config.avg_timing_iterations = 8
        # if config.use_fp16:
        #     builder_config.set_flag(trt.BuilderFlag.FP16)
        builder.fp16_mode = True  # Set the builder to FP16 mode
        builder.max_batch_size = 1  # Set the maximum batch size
        builder.max_workspace_size = 1 << 30  # Set the maximum workspace size
        with open('model_data/model.onnx', 'rb') as model_file:  # Load the ONNX model
            parser.parse(model_file.read())
        engine = builder.build_cuda_engine(network)  # Build the TensorRT engine

    # Serialize the engine to a file
    with open('model_data/model_fp16.engine', 'wb') as f:
        f.write(engine.serialize())
fp16_2()


def infer(onnx_path, engine_path):
    # 导入必用依赖
    import tensorrt as trt
    # 创建logger：日志记录器
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建构建器builder
    builder = trt.Builder(logger)
    # 预创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 加载onnx解析器
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        pass  # Error handling code here
    # builder配置
    config = builder.create_builder_config()
    # 分配显存作为工作区间，一般建议为显存一半的大小
    config.max_workspace_size = 1 << 30  # 1 Mi
    serialized_engine = builder.build_serialized_network(network, config)   # 只是序列化网络，没有构造引擎
    # 序列化生成engine文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        print("generate file success!")