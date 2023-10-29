use candle::{Tensor, Module, Result, Shape, Device, DType, D, IndexOp};
use candle_nn::{Activation, VarBuilder, Conv2d, conv2d, Conv2dConfig};
use crate::models::clip::configuration_clip::{CLIPTextConfig, CLIPVisionConfig, CLIPConfig};
use crate::models::with_tracing::{Embedding, Linear, linear};

struct CLIPVisionModelOutput {
    image_embeds: Option<Tensor>,
    last_hidden_state: Tensor,
    hidden_states: Option<Vec<Tensor>>,
    attentions: Option<Vec<Tensor>> 
}

struct CLIPTextModelOutput {
    text_embeds: Option<Tensor>,
    last_hidden_state: Tensor,
    hidden_states: Option<Vec<Tensor>>,
    attentions: Option<Vec<Tensor>>
}

struct CLIPOutput {
    loss: Option<Tensor>,
    logits_per_image: Tensor,
    logits_per_text: Tensor, 
    text_embeds: Tensor,
    image_embeds: Tensor, 
    text_model_output: CLIPTextModelOutput,
    vision_model_output: CLIPVisionModelOutput
}


struct CLIPVisionEmbeddings {
    embed_dim: usize,
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    position_ids: Tensor
}

impl CLIPVisionEmbeddings {
    fn load(vb: VarBuilder, config: &CLIPVisionConfig) -> Result<Self> {
        let num_patches = usize::pow(config.image_size / config.patch_size, 2);
        let num_positions = num_patches + 1;
        let position_ids = Tensor::arange(0, num_positions as u32, &Device::Cpu)?.expand(Shape::from_dims(&[1, num_positions]))?;
        let patch_embedding = conv2d(config.num_channels, config.hidden_size, config.patch_size, Conv2dConfig::default(), vb.pp("clip_vision_patch_embedding"))?;
        let position_embedding = Embedding::new(num_positions, config.hidden_size, vb.pp("clip_vision_position_embedding")
        )?;
        Ok(Self { 
            embed_dim: config.hidden_size,
            class_embedding: Tensor::randn(0f32, 1f32, (config.hidden_size, ), &Device::Cpu)?, 
            patch_embedding: patch_embedding, 
            position_embedding: position_embedding, 
            position_ids: position_ids }
        )
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.shape().dims()[0];
        let target_dtype = self.patch_embedding.weight().dtype();
        let mut patch_embeds = self.patch_embedding.forward(&pixel_values.to_dtype(target_dtype)?)?;
        patch_embeds = patch_embeds.flatten(2, 2)?.transpose(1, 2)?;
        
        let class_embeds = self.class_embedding.expand(Shape::from_dims(&[batch_size, 1, self.embed_dim]))?;
        let mut embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        embeddings = (embeddings + self.position_embedding.forward(&self.position_ids)?)?;
        Ok(embeddings)
    }
}


struct CLIPTextEmbeddings {
    embed_dim: usize,
    token_embedding: Embedding,
    position_embedding: Embedding,
    position_ids: Tensor
}


impl CLIPTextEmbeddings {
    fn load(vb: VarBuilder, config: &CLIPTextConfig) -> Result<Self> {
        let token_embedding = Embedding::new(config.vocab_size, config.hidden_size, vb.pp("token_embedding"))?;
        let position_embedding = Embedding::new(config.max_position_embeddings, config.hidden_size, vb.pp("text_position_embedding"))?;
        let position_ids = Tensor::arange(0, config.max_position_embeddings as u32, &Device::Cpu)?.expand(Shape::from_dims(&[1, config.max_position_embeddings]))?;

        Ok(Self{
            embed_dim: config.hidden_size,
            token_embedding: token_embedding, 
            position_embedding: position_embedding, 
            position_ids: position_ids
        })
    }

    fn forward(&self, input_ids: Option<&Tensor>, position_ids: Option<&Tensor>, input_embeds: Option<&Tensor>) -> Result<Tensor> {

        let seq_length = match input_ids {
            Some(ids) => ids.shape().dims()[ids.rank() - 1],
            None => input_embeds.unwrap().shape().dims()[input_embeds.unwrap().rank() - 2]
        };

        let position_ids = match position_ids {
            Some(ids) => ids.to_owned(),
            None => self.position_ids.i((.., ..seq_length) )?
        };
    

        let input_embeds = match input_embeds {
            Some(embeds) => embeds.to_owned(),
            None => self.token_embedding.forward(&input_ids.unwrap())?
        };

        let position_embeddings = self.position_embedding.forward(&position_ids)?;

        let embeddings = (input_embeds + position_embeddings)?;

        Ok(embeddings)

    }

}


struct CLIPAttention {
    embed_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear
}

impl CLIPAttention {
    fn load(vb: VarBuilder, config: &CLIPTextConfig) -> Result<Self> {
        Ok(CLIPAttention { 
            embed_dim: config.hidden_size, 
            q_proj: linear(config.hidden_size, config.hidden_size, vb.pp("clip_query_proj"))?,
            k_proj: linear(config.hidden_size, config.hidden_size, vb.pp("clip_key_proj"))?,
            v_proj: linear(config.hidden_size, config.hidden_size, vb.pp("clip_value_proj"))?,
            out_proj: linear(config.hidden_size, config.hidden_size, vb.pp("clip_out_proj"))?,
        })
    }


}
struct CLIPMLP {}
struct CLIPEncoderLayer {}

struct CLIPEncoder {}

struct CLIPTextTransformer {}

struct CLIPTextModel {}

struct CLIPVisionTransformer {}

struct CLIPVisionModel {}



