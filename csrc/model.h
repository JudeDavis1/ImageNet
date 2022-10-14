#include <iostream>
#include <torch/torch.h>

namespace nn = torch::nn;


#define CHANNELS 3
#define LATENT_SPACE 256


// Base for all models
class BaseModel : public nn::Module
{
public:
    void save(std::string& path);
    void load(std::string& path);
    
    virtual void Block(int prev, int nfg, int stride=2, int padding=1, bool batch_norm=true);
};


// The model we try to train
class Generator : public BaseModel
{
public:
    Generator();

    torch::Tensor forward(torch::Tensor x);

    ~Generator();
private:
    nn::Sequential m_model;
};

// The model that corrects the Generator
class Discriminator : public BaseModel
{
public:
    Discriminator();

    torch::Tensor forward(torch::Tensor x);

    ~Discriminator();
private:
    nn::Sequential m_model;
}




// TORCH_MODULE_IMPL(Generator, GeneratorImpl);


