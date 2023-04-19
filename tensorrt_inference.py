from typing import Union, Optional, Sequence, Dict, Any

import torch
import tensorrt as trt


class TRTWrapper(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):   # 判断引擎文件是否为字符串类型，是则需要反序列化引擎
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:   # 创建一个日志和运行时对象，用于反序列化引擎
                with open(self.engine, mode='rb') as f:   # 读取二进制引擎文件
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)   # 将引擎文件反序列化为ICudaEngine对象
        self.context = self.engine.create_execution_context()   # 创建一个执行上下文对象，用于执行推理

        names = [_ for _ in self.engine]   # 获取所有输入和输出的名字
        input_names = list(filter(self.engine.binding_is_input, names))   # 筛选出所有的输入名字
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None

        bindings = [None] * (len(self._input_names) + len(self._output_names))   # 用于存储输入张量和输出张量的指针
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)   # 获取输入张量再profile中的形状
            # assert input_tensor.dim() == len(profile[0]), 'Input dim is different from engine profile.'

            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)   # 获取在binding列表中的索引

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()   # 强拷贝
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)   # 异步推理
        return outputs


model = TRTWrapper('model_data/model.engine', ['output'])
output = model(dict(input=torch.randn(1, 3, 224, 224).cuda()))
print(output)



