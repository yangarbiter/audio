#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <torch/script.h>
#include "rnnt.h"

int64_t cpu_rnnt_loss(torch::Tensor acts,
                      torch::Tensor labels,
                      torch::Tensor input_lengths,
                      torch::Tensor label_lengths,
                      torch::Tensor costs,
                      torch::Tensor grads,
                      int64_t blank_label,
                      int64_t num_threads) {

    int maxT = acts.size(1);
    int maxU = acts.size(2);
    int minibatch_size = acts.size(0);
    int alphabet_size = acts.size(3);

    rnntOptions options;
    memset(&options, 0, sizeof(options));
    options.maxT = maxT;
    options.maxU = maxU;
    options.blank_label = blank_label;
    options.batch_first = true;
    options.loc = RNNT_CPU;
    options.num_threads = num_threads;

    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);

    size_t cpu_size_bytes = 0;
    switch (acts.scalar_type()) {
      case torch::ScalarType::Float:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                           false, &cpu_size_bytes);

        std::vector<float> cpu_workspace(cpu_size_bytes / sizeof(float), 0);

        compute_rnnt_loss(acts.data<float>(), grads.data<float>(),
                          labels.data<int>(), label_lengths.data<int>(),
                          input_lengths.data<int>(), alphabet_size,
                          minibatch_size, costs.data<float>(),
                          cpu_workspace.data(), options);

        return 0;
        }
      case torch::ScalarType::Double:
        {
        get_workspace_size(maxT, maxU, minibatch_size,
                           false, &cpu_size_bytes,
                           sizeof(double));

        std::vector<double> cpu_workspace(cpu_size_bytes / sizeof(double), 0);

        compute_rnnt_loss_fp64(acts.data<double>(), grads.data<double>(),
                               labels.data<int>(), label_lengths.data<int>(),
                               input_lengths.data<int>(), alphabet_size,
                               minibatch_size, costs.data<double>(),
                               cpu_workspace.data(), options);

        return 0;
        }
      default:
        TORCH_CHECK(false,
            std::string(__func__) + " not implemented for '" + toString(acts.scalar_type()) + "'"
        );
    }
    return -1;
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
    m.impl("rnnt_loss", &cpu_rnnt_loss);
}
