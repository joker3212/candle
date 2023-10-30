use crate::models::clip::configuration_clip::{CLIPConfig, CLIPTextConfig, CLIPVisionConfig};
use crate::models::with_tracing::{linear, Embedding, Linear};
//use crate::quantized_var_builder::VarBuilder;
use candle::{DType, Device, IndexOp, Module, Result, Shape, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{
    conv2d_no_bias, layer_norm, Activation, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig,
    VarBuilder,
};

struct CLIPVisionModelOutput {
    image_embeds: Option<Tensor>,
    last_hidden_state: Tensor,
    hidden_states: Option<Vec<Tensor>>,
    attentions: Option<Vec<Tensor>>,
}

struct CLIPTextModelOutput {
    text_embeds: Option<Tensor>,
    last_hidden_state: Tensor,
    hidden_states: Option<Vec<Tensor>>,
    attentions: Option<Vec<Tensor>>,
}

struct CLIPOutput {
    loss: Option<Tensor>,
    logits_per_image: Tensor,
    logits_per_text: Tensor,
    text_embeds: Tensor,
    image_embeds: Tensor,
    text_model_output: CLIPTextModelOutput,
    vision_model_output: CLIPVisionModelOutput,
}

struct CLIPVisionEmbeddings {
    embed_dim: usize,
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    position_ids: Tensor,
}

impl CLIPVisionEmbeddings {
    fn load(vb: VarBuilder, config: &CLIPVisionConfig) -> Result<Self> {
        let num_patches = usize::pow(config.image_size / config.patch_size, 2);
        let num_positions = num_patches + 1;
        let position_ids = Tensor::arange(0, num_positions as u32, &Device::Cpu)?
            .expand(Shape::from_dims(&[1, num_positions]))?;
        let patch_embedding = conv2d_no_bias(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            Conv2dConfig::default(),
            vb.pp("clip_vision_patch_embedding"),
        )?;
        let position_embedding = Embedding::new(
            num_positions,
            config.hidden_size,
            vb.pp("clip_vision_position_embedding"),
        )?;
        Ok(Self {
            embed_dim: config.hidden_size,
            class_embedding: Tensor::randn(0f32, 1f32, (config.hidden_size,), &Device::Cpu)?,
            patch_embedding: patch_embedding,
            position_embedding: position_embedding,
            position_ids: position_ids,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.shape().dims()[0];
        let target_dtype = self.patch_embedding.weight().dtype();
        let mut patch_embeds = self
            .patch_embedding
            .forward(&pixel_values.to_dtype(target_dtype)?)?;
        patch_embeds = patch_embeds.flatten(2, 2)?.transpose(1, 2)?;

        let class_embeds =
            self.class_embedding
                .expand(Shape::from_dims(&[batch_size, 1, self.embed_dim]))?;
        let mut embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        embeddings = (embeddings + self.position_embedding.forward(&self.position_ids)?)?;
        Ok(embeddings)
    }
}

struct CLIPTextEmbeddings {
    embed_dim: usize,
    token_embedding: Embedding,
    position_embedding: Embedding,
    position_ids: Tensor,
}

impl CLIPTextEmbeddings {
    fn load(vb: VarBuilder, config: &CLIPTextConfig) -> Result<Self> {
        let token_embedding = Embedding::new(
            config.vocab_size,
            config.hidden_size,
            vb.pp("token_embedding"),
        )?;
        let position_embedding = Embedding::new(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("text_position_embedding"),
        )?;
        let position_ids = Tensor::arange(0, config.max_position_embeddings as u32, &Device::Cpu)?
            .expand(Shape::from_dims(&[1, config.max_position_embeddings]))?;

        Ok(Self {
            embed_dim: config.hidden_size,
            token_embedding: token_embedding,
            position_embedding: position_embedding,
            position_ids: position_ids,
        })
    }

    fn forward(
        &self,
        input_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_length = match input_ids {
            Some(ids) => ids.shape().dims()[ids.rank() - 1],
            None => input_embeds.unwrap().shape().dims()[input_embeds.unwrap().rank() - 2],
        };

        let position_ids = match position_ids {
            Some(ids) => ids.to_owned(),
            None => self.position_ids.i((.., ..seq_length))?,
        };

        let input_embeds = match input_embeds {
            Some(embeds) => embeds.to_owned(),
            None => self.token_embedding.forward(&input_ids.unwrap())?,
        };

        let position_embeddings = self.position_embedding.forward(&position_ids)?;

        let embeddings = (input_embeds + position_embeddings)?;

        Ok(embeddings)
    }
}

struct CLIPAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl CLIPAttention {
    fn load(vb: VarBuilder, config: &CLIPTextConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let scale: f32 = 1f32 / (head_dim as f32).sqrt();
        Ok(CLIPAttention {
            embed_dim: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            scale: scale,
            q_proj: linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("clip_query_proj"),
            )?,
            k_proj: linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("clip_key_proj"),
            )?,
            v_proj: linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("clip_value_proj"),
            )?,
            out_proj: linear(
                config.hidden_size,
                config.hidden_size,
                vb.pp("clip_out_proj"),
            )?,
        })
    }

    fn _shape(&self, tensor: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        Ok(tensor
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (bsz, tgt_len, embed_dim) = hidden_states.dims3()?;
        let scale_tensor = Tensor::from_vec([self.scale].to_vec(), (1,), &Device::Cpu)?;
        // get query proj
        let mut query_states = self
            .q_proj
            .forward(hidden_states)?
            .broadcast_mul(&scale_tensor)?;
        let mut key_states = self._shape(&self.k_proj.forward(hidden_states)?, tgt_len, bsz)?;
        let mut value_states = self._shape(&self.v_proj.forward(hidden_states)?, tgt_len, bsz)?;

        let proj_shape = (bsz * self.num_heads, tgt_len, self.head_dim);
        query_states = self
            ._shape(&query_states, tgt_len, bsz)?
            .reshape(proj_shape)?;

        key_states = key_states.reshape(proj_shape)?;
        value_states = value_states.reshape(proj_shape)?;

        let mut attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        attn_weights = softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        Ok(attn_output)
    }
}

struct CLIPMLP {
    activation_fn: Activation,
    fc1: Linear,
    fc2: Linear,
}

impl CLIPMLP {
    fn load(vb: VarBuilder, config: &CLIPTextConfig) -> Result<Self> {
        Ok(Self {
            activation_fn: config.hidden_act,
            fc1: linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut output = self.fc1.forward(hidden_states)?;
        output = self.activation_fn.forward(&output)?;
        output = self.fc2.forward(&output)?;
        Ok(output)
    }
}
struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layer_norm1: LayerNorm,
    mlp: CLIPMLP,
    layer_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    fn load(vb: VarBuilder, config: &CLIPTextConfig) -> Result<Self> {
        Ok(Self {
            self_attn: CLIPAttention::load(vb.pp("self_attention"), &config)?,
            layer_norm1: layer_norm(
                config.hidden_size,
                LayerNormConfig::default(),
                vb.pp("layer_norm1"),
            )?,
            mlp: CLIPMLP::load(vb.pp("mlp"), &config)?,
            layer_norm2: layer_norm(
                config.hidden_size,
                LayerNormConfig::default(),
                vb.pp("layer_norm2"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let layer_norm1_output = self.layer_norm1.forward(&hidden_states)?;
        let attn_output = self.self_attn.forward(&layer_norm1_output)?;
        let layer1_out = (hidden_states + attn_output)?;

        let layer_norm2_output = self.layer_norm2.forward(&layer1_out)?;
        let mlp_output = self.mlp.forward(&layer_norm2_output)?;
        let output = (layer1_out + mlp_output)?;

        Ok(output)
    }
}

struct CLIPEncoder {
    layers: Vec<CLIPEncoderLayer>,
}

impl CLIPEncoder {
    fn load(vb: VarBuilder, config: &CLIPConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 1..config.clip_text_config.num_hidden_layers {
            layers.push(CLIPEncoder::load(
                vb.pp("encoder_layer"),
                &config.clip_text_config,
            )?);
        }
        Ok(Self { layers: layers })
    }

    fn forward(&self, input_embeds: &Tensor) -> Result<Tensor> {
        let mut layer_output = input_embeds;
        for i in 1..self.layers.len() {
            layer_output = self.layers.forward(layer_output)?;
        }
        Ok(layer_output)
    }
}

struct CLIPTextTransformer {
    embed_dim: usize,
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder,
    final_layer_norm: LayerNorm,
    eos_token_id: usize,
}

struct CLIPTextModel {}

struct CLIPVisionTransformer {}

struct CLIPVisionModel {}
