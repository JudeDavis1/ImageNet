#include "model.h"


void NO_IMPL_ERR() { throw std::runtime_error("This function has not been implemented yet."); }



Generator::Generator()
{
    this->m_model = register_module("model", nn::Sequential());

    int nfg = 64;
    this->Block(LATENT_SPACE, nfg * 4, 1, 0);
    this->Block(nfg * 4, nfg * 2);
    this->Block(nfg * 2, nfg);
    this->m_model->push_back(nn::ConvTranspose2d(
        nn::ConvTranspose2dOptions(nfg, CHANNELS, 4)
        .stride(2)
        .bias(false)
    ));
}

torch::Tensor Generator::forward(torch::Tensor x)
{
    return this->m_model->forward(x);
}

void Generator::Block(int prev, int nfg, int stride, int padding, bool batch_norm)
{
    this->m_model->push_back(
        nn::ConvTranspose2d(
            nn::ConvTranspose2dOptions(prev, nfg, 4)
                .stride(stride)
                .padding(padding)
                .bias(false)
        )
    );

    if (batch_norm)
        this->m_model->push_back(nn::BatchNorm2d(nfg));

    this->m_model->push_back(nn::ReLU(true));
}

Generator::~Generator() {}



// Discriminator

Discriminator::Discriminator()
{
    // Create model
    this->m_model = ;
}




// Base Model

void BaseModel::save(std::string& path)
{
    NO_IMPL_ERR();
}

void BaseModel::load(std::string& path)
{
    NO_IMPL_ERR();
}




