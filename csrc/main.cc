#include <iostream>

// Local libraries
#include "model.h"


int main()
{
    auto gen = std::make_shared<GeneratorImpl>();
    torch::Tensor result = gen->forward(torch::randn({1, LATENT_SPACE, 1, 1}));

    std::cout << result.sizes() << std::endl;
}
