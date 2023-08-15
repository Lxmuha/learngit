#include <tuple>
#include <ATen/ATen.h>


::std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenSwaiType::native_layer_norm(
    const at::Tensor& x, at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt, const c10::optional<at::Tensor>& bias_opt,
    double eps)
{

    const c10::OptionalDeviceGuard device_guard(at::device_of(x));

    TORCH_CHECK(weight_opt.has_value() && bias_opt.has_value(),
        "native_layer_norm weight or bias null");

    auto input = *(x.expect_contiguous());
    auto weight = c10::make_optional(*(weight_opt.value().expect_contiguous()));
    auto bias = c10::make_optional(*(bias_opt.value().expect_contiguous()));

    at::TensorArg input_arg{input, "input", 1}, weight_arg{ *weight, "weight", 2 },
        bias_arg{ *bias, "bias", 3 };
    at::CheckedFrom c = "native_layer_norm";
    at::checkAllSameType(c, { input_arg, weight_arg, bias_arg });
    TORCH_CHECK(input.dtype() == at::kFloat || input.dtype() == at::kHalf,
        "native_layer_norm only support half and float");
    const auto input_shape = input.sizes();
    const int axis = input.dim() - normalized_shape.size();
    std::vector<int64_t> stat_shape;
    for (size_t idx = 0; idx < axis; ++idx) {
        stat_shape.push_back(input_shape[idx]);
    }
    for (size_t idx = axis; idx < input.dim(); ++idx) {
        stat_shape.push_back(1);
    }

    at::Tensor output = at::empty(input_shape, input.options());
    at::Tensor mean = at::empty(stat_shape, input.options());
    at::Tensor invstd = at::empty(stat_shape, input.options());

    std::vector<at::Tensor> swai_attensors = { input, output, mean, invstd };
    auto swaitens = bridge::GetSwaiTensors(swai_attensors);
    std::vector<c10::optional<at::Tensor>> swai_opt_attensors = { weight, bias };
    auto swaitens_opt = bridge::GetOptSwaiTensors(swai_opt_attensors);
    for (size_t i = 0; i < swaitens.size(); ++i) {
        swaitens[i].requestSwai();
    }
    for (size_t i = 0; i < swaitens_opt.size(); ++i) {
        if (swaitens_opt[i].has_value()) {
            swaitens_opt[i]->requestSwai();
        }
    }

    swdnn::native_layer_norm(swaitens[0], normalized_shape, swaitens_opt[0], swaitens_opt[1], eps,
        swaitens[1], swaitens[2], swaitens[3]);
    return ::std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, mean, invstd);
}


void native_layer_norm(const SWAITensor& input, at::IntArrayRef normalized_shape,
    const c10::optional<SWAITensor>& weight,
    const c10::optional<SWAITensor>& bias, double eps, SWAITensor& output,
    SWAITensor& mean, SWAITensor& invstd) {
    /* extract meta data */
    auto input_cpu = *input.CurrentTensorData();
    auto mean_cpu = *mean.CurrentTensorData();
    // size_t x_size = input_cpu.nbytes();
    // size_t stat_size = mean_cpu.nbytes();
    auto input_shape = input_cpu.sizes();
    const int axis = input_cpu.dim() - normalized_shape.size();
    int64_t M = c10::multiply_integers(input_shape.cbegin(), input_shape.cbegin() + axis);
    int64_t N = c10::multiply_integers(input_shape.cbegin() + axis, input_shape.cend());

    /* device input */
    void* swai_x_data = input.swai_data();
    void* swai_weight_data = weight->swai_data();
    void* swai_bias_data = bias->swai_data();

    /* device result */
    void* swai_y_data = output.swai_data();
    //   sdaaMemset(swai_y_data, 0, x_size);
    void* swai_mean_data = mean.swai_data();
    //   sdaaMemset(swai_mean_data, 0, stat_size);
    void* swai_invstd_data = invstd.swai_data();
    //   sdaaMemset(swai_invstd_data, 0, stat_size);

    runLayerNormForward(swai_x_data, swai_weight_data, swai_bias_data, swai_y_data, swai_mean_data,
        swai_invstd_data, M, N, eps, getSwdnnDataType(input_cpu));
}



void runLayerNormForward(const void* x, const void* gamma, const void* beta, void* y, void* mean,
    void* rstd, int64_t row, int64_t col, double eps,
    swdnnDataType_t dataType) {
    int N = row;
    int C = col;
    int H = 1;
    int W = 1;

    int x_N = N, x_C = C, x_H = H, x_W = W;
    int y_N = N, y_C = C, y_H = H, y_W = W;
    int mean_N = N, mean_C = 1, mean_H = 1, mean_W = 1;
    int rstd_N = N, rstd_C = 1, rstd_H = 1, rstd_W = 1;
    int gamma_N = 1, gamma_C = C, gamma_H = H, gamma_W = W;
    int beta_N = 1, beta_C = C, beta_H = H, beta_W = W;

    swdnnHandle_t swdnnHandle = getSwdnnHandle();
    swdnnTensorDescriptor_t x_Desc, y_Desc, mean_Desc, rstd_Desc, gamma_Desc, beta_Desc;
    swdnnLayerMode_t lnMode = SWDNN_LAYER_NORM_0;

    SWAI_CALLDNN(swdnnCreateTensorDescriptor(&x_Desc));
    SWAI_CALLDNN(swdnnCreateTensorDescriptor(&y_Desc));
    SWAI_CALLDNN(swdnnCreateTensorDescriptor(&mean_Desc));
    SWAI_CALLDNN(swdnnCreateTensorDescriptor(&rstd_Desc));
    SWAI_CALLDNN(swdnnCreateTensorDescriptor(&gamma_Desc));
    SWAI_CALLDNN(swdnnCreateTensorDescriptor(&beta_Desc));

    SWAI_CALLDNN(swdnnSetTensor4dDescriptor(x_Desc, SWDNN_TENSOR_NHWC, dataType, x_N, x_C, x_H, x_W));
    SWAI_CALLDNN(swdnnSetTensor4dDescriptor(y_Desc, SWDNN_TENSOR_NHWC, dataType, y_N, y_C, y_H, y_W));
    SWAI_CALLDNN(swdnnSetTensor4dDescriptor(mean_Desc, SWDNN_TENSOR_NHWC, dataType, mean_N, mean_C,
        mean_H, mean_W));
    SWAI_CALLDNN(swdnnSetTensor4dDescriptor(rstd_Desc, SWDNN_TENSOR_NHWC, dataType, rstd_N, rstd_C,
        rstd_H, rstd_W));
    SWAI_CALLDNN(swdnnSetTensor4dDescriptor(gamma_Desc, SWDNN_TENSOR_NHWC, dataType, gamma_N, gamma_C,
        gamma_H, gamma_W));
    SWAI_CALLDNN(swdnnSetTensor4dDescriptor(beta_Desc, SWDNN_TENSOR_NHWC, dataType, beta_N, beta_C,
        beta_H, beta_W));

    SWAI_RECDDNN(swdnnLayerNormForward(swdnnHandle, lnMode, x_Desc, x, gamma_Desc, gamma, beta_Desc,
        beta, eps, y_Desc, y, mean_Desc, mean, rstd_Desc, rstd));

    SWAI_CALLDNN(swdnnDestroyTensorDescriptor(x_Desc));
    SWAI_CALLDNN(swdnnDestroyTensorDescriptor(y_Desc));
    SWAI_CALLDNN(swdnnDestroyTensorDescriptor(mean_Desc));
    SWAI_CALLDNN(swdnnDestroyTensorDescriptor(rstd_Desc));
    SWAI_CALLDNN(swdnnDestroyTensorDescriptor(gamma_Desc));
    SWAI_CALLDNN(swdnnDestroyTensorDescriptor(beta_Desc));
}