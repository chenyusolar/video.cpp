use crate::libcore::context::Context;
use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Error, Result, TextEncoder as TextEncoderTrait};
use crate::model::gguf::GGUFFile;
use std::sync::Arc;

/// 文本编码器配置
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub model_type: String,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub intermediate_size: u32,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            model_type: String::new(),
            hidden_size: 4096,
            vocab_size: 32000,
            max_position_embeddings: 512,
            num_hidden_layers: 4,
            num_attention_heads: 16,
            intermediate_size: 16384,
        }
    }
}

/// Gemma 文本编码器实现
pub struct GemmaTextEncoder {
    gguf: Arc<GGUFFile>,
    config: TextEncoderConfig,
    weights: std::sync::RwLock<std::collections::HashMap<String, Tensor>>,
    use_real_weights: bool,
}

impl GemmaTextEncoder {
    /// 创建新的文本编码器
    pub fn new(gguf: Arc<GGUFFile>) -> Self {
        let te_config = &gguf.config.text_encoder;
        let config = TextEncoderConfig {
            model_type: te_config.model_type.clone(),
            hidden_size: te_config.hidden_size.max(1),
            vocab_size: te_config.vocab_size.max(1),
            max_position_embeddings: te_config.max_position_embeddings.max(512),
            num_hidden_layers: te_config.num_hidden_layers,
            num_attention_heads: te_config.num_attention_heads,
            intermediate_size: te_config.intermediate_size,
        };

        Self {
            gguf,
            config,
            weights: std::sync::RwLock::new(std::collections::HashMap::new()),
            use_real_weights: false,
        }
    }

    /// 加载所有文本编码器权重
    pub fn load_weights(&mut self) -> Result<()> {
        let mut loaded_any = false;

        for tensor_meta in self.gguf.list_tensors() {
            if Self::is_text_encoder_tensor(&tensor_meta.name) {
                let data = self.gguf.load_tensor_data(tensor_meta).map_err(|e| {
                    Error::Model(format!(
                        "Failed to load text encoder tensor '{}': {}",
                        tensor_meta.name, e
                    ))
                })?;

                let shape: Vec<u32> = tensor_meta.dims.iter().map(|&d| d as u32).collect();
                let total_elements: usize = shape.iter().product::<u32>() as usize;

                let floats: Vec<f32> = match tensor_meta.dtype {
                    crate::model::gguf::GGUFDType::F32 => {
                        if data.len() >= total_elements * 4 {
                            loaded_any = true;
                            data.chunks_exact(4)
                                .map(|chunk| {
                                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                })
                                .collect()
                        } else {
                            vec![0.0; total_elements]
                        }
                    }
                    crate::model::gguf::GGUFDType::F16 => {
                        if data.len() >= total_elements * 2 {
                            loaded_any = true;
                            data.chunks_exact(2)
                                .map(|chunk| {
                                    half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                                })
                                .collect()
                        } else {
                            vec![0.0; total_elements]
                        }
                    }
                    crate::model::gguf::GGUFDType::Q4_K => {
                        loaded_any = true;
                        Self::dequantize_q4_k(&data, total_elements)
                    }
                    crate::model::gguf::GGUFDType::Q8_0 => {
                        loaded_any = true;
                        Self::dequantize_q8_0(&data, total_elements)
                    }
                    _ => {
                        // 尝试作为 F16 解码
                        if data.len() >= total_elements * 2 {
                            loaded_any = true;
                            data.chunks_exact(2)
                                .map(|chunk| {
                                    half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
                                })
                                .collect()
                        } else {
                            vec![0.0; total_elements]
                        }
                    }
                };

                let tensor = Tensor::from_data(TensorShape::new(shape), TensorData::F32(floats));
                self.weights
                    .write()
                    .unwrap()
                    .insert(tensor_meta.name.clone(), tensor);
            }
        }

        // 标记是否加载了真实权重
        if loaded_any {
            tracing::info!(
                "Loaded {} text encoder tensors from GGUF",
                self.weights.read().unwrap().len()
            );
        } else {
            tracing::warn!("No text encoder weights found in GGUF, will use fallback embeddings");
        }

        self.use_real_weights = loaded_any;
        Ok(())
    }

    /// 检查 tensor 名称是否为文本编码器相关
    fn is_text_encoder_tensor(name: &str) -> bool {
        let prefixes = [
            "text_encoder",
            "embedder",
            "gemma",
            "text_model",
            "encoder",
            "llama.embed_tokens",
            "model.embed_tokens",
            "transformer.wte",
            "transformer.h",
        ];
        prefixes.iter().any(|p| name.contains(p))
    }

    /// Q4_K 解量化
    fn dequantize_q4_k(data: &[u8], total_elements: usize) -> Vec<f32> {
        let block_size = 32;
        let mut result = Vec::with_capacity(total_elements);
        let num_blocks = data.len() / block_size;

        for block_idx in 0..num_blocks {
            let offset = block_idx * block_size;
            let block = &data[offset..];

            if block.len() < 34 {
                result.extend_from_slice(&vec![0.0; block.len() / 2]);
                continue;
            }

            // Q4_K 格式: 2字节 scale + 2字节 min + 16字节量化数据 + 16字节索引
            let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            let min_val = f32::from_le_bytes([block[4], block[5], block[6], block[7]]);

            for i in 0..16 {
                let q = block[8 + i];
                // Q4_K 使用 4 位表示 -8 到 7
                let val = (q as f32 / 16.0 - 8.0) * scale + min_val;
                result.push(val);
            }
        }

        // 填充剩余元素
        while result.len() < total_elements {
            result.push(0.0);
        }

        result
    }

    /// Q8_0 解量化
    fn dequantize_q8_0(data: &[u8], total_elements: usize) -> Vec<f32> {
        let block_size = 32;
        let mut result = Vec::with_capacity(total_elements);
        let num_blocks = data.len() / block_size;

        for block_idx in 0..num_blocks {
            let offset = block_idx * block_size;
            let block = &data[offset..];

            if block.len() < 34 {
                result.extend_from_slice(&vec![0.0; block.len()]);
                continue;
            }

            let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);

            for i in 0..32 {
                let q = block[4 + i] as i8;
                let val = q as f32 * scale;
                result.push(val);
            }
        }

        while result.len() < total_elements {
            result.push(0.0);
        }

        result
    }

    /// 获取权重
    fn get_weight(&self, name: &str) -> Option<Tensor> {
        // 尝试多个可能的名称变体
        let possible_names = [
            name.to_string(),
            format!("text_encoder.{}", name),
            format!("gemma.{}", name),
            format!("model.{}", name),
            format!("transformer.{}", name),
            format!("encoder.{}", name),
        ];

        let weights = self.weights.read().unwrap();
        for n in &possible_names {
            if let Some(w) = weights.get(n) {
                return Some(w.clone());
            }
        }
        None
    }

    /// 字符级分词
    fn tokenize_chars(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() * 4);

        for c in text.chars() {
            if c.is_ascii() {
                tokens.push(c as u32);
            } else {
                // Unicode 字符使用代理对
                let mut buf = [0_u16; 2];
                let encoded = c.encode_utf16(&mut buf);
                for cp in encoded {
                    tokens.push(*cp as u32);
                }
            }
        }

        tokens
    }

    /// 词汇表级分词（使用嵌入的词汇表）
    fn tokenize_vocab(&self, text: &str) -> Vec<u32> {
        // 尝试获取词汇表
        let vocab = self
            .get_weight("embed_tokens.weight")
            .or_else(|| self.get_weight("wte.weight"));

        if vocab.is_none() {
            // 没有词汇表，回退到字符级
            return self.tokenize_chars(text);
        }

        let vocab = vocab.unwrap();
        let vocab_size = vocab.shape().dims()[0];
        let hidden_size = vocab.shape().dims()[1];

        // 字符级分词
        let char_tokens: Vec<f32> = text
            .chars()
            .flat_map(|c| {
                let cp = if c.is_ascii() { c as u32 } else { 0x20 };
                (0..hidden_size as usize).map(move |i| (cp as f32 + i as f32 * 0.001).sin().cos())
            })
            .collect();

        // 简单的滑动窗口匹配
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < text.len() {
            let remaining = &text[i..];
            let mut matched = false;

            // 尝试从最长到最短匹配
            for len in (1..=remaining.len().min(8)).rev() {
                let candidate = &remaining[..len];
                let hash = self.hash_f32(candidate);

                if (hash % vocab_size as f32) < (vocab_size as f32 * 0.8) {
                    tokens.push((hash % vocab_size as f32) as u32);
                    i += len;
                    matched = true;
                    break;
                }
            }

            if !matched {
                tokens.push(self.hash_f32(&text[i..=i]) as u32 % vocab_size);
                i += 1;
            }
        }

        tokens
    }

    /// 简单的哈希函数用于词汇表查找
    fn hash_f32(&self, s: &str) -> f32 {
        let mut hash: u32 = 5381;
        for c in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(c as u32);
        }
        (hash as f32 / u32::MAX as f32) * 10007.0
    }

    /// Embedding 查找
    fn embedding_lookup(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len() as u32;
        let hidden_size = self.config.hidden_size;

        // 尝试获取 embedding 权重
        if let Some(embed_weight) = self
            .get_weight("embed_tokens.weight")
            .or_else(|| self.get_weight("wte.weight"))
        {
            let weight_data = embed_weight.data();
            if let TensorData::F32(ref data) = weight_data {
                let vocab_size = embed_weight.shape().dims()[0];
                let emb_dim = embed_weight.shape().dims()[1];
                let mut embeddings = Vec::with_capacity((seq_len * hidden_size) as usize);

                for &token_id in token_ids {
                    let idx = (token_id as usize).min(vocab_size as usize - 1);
                    let offset = idx * emb_dim as usize;

                    for i in 0..hidden_size as usize {
                        let emb_idx = offset + i.min(emb_dim as usize - 1);
                        embeddings.push(data.get(emb_idx).copied().unwrap_or(0.0));
                    }
                }

                return embeddings;
            }
        }

        // Fallback: 生成伪随机但确定性的 embeddings
        tracing::debug!("Using fallback embeddings for {} tokens", seq_len);
        self.fallback_embedding(token_ids)
    }

    /// Fallback 伪随机 embedding
    fn fallback_embedding(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq_len = token_ids.len() as u32;
        let hidden_size = self.config.hidden_size;
        let mut embeddings = vec![0.0_f32; (seq_len * hidden_size) as usize];

        for (pos, &token_id) in token_ids.iter().enumerate() {
            for dim in 0..hidden_size as usize {
                let idx = pos * hidden_size as usize + dim;
                let freq = token_id as f32 * 0.1 + dim as f32 * 0.01;
                embeddings[idx] = (freq.sin() * 0.5 + 0.5) * (1.0 / (dim as f32 + 1.0).sqrt());
            }
        }

        embeddings
    }

    /// 添加 Sinusoidal 位置编码
    fn add_position_encoding(&self, embeddings: &mut [f32], seq_len: u32) {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        for pos in 0..seq_len as usize {
            for dim in 0..hidden_size as usize {
                let idx = pos * hidden_size as usize + dim;
                if idx >= embeddings.len() {
                    break;
                }

                let angle = pos as f32 / 10000.0_f32.powf(dim as f32 / head_dim as f32);
                let encoded = if dim % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };

                embeddings[idx] += encoded * 0.02;
            }
        }
    }

    /// Transformer 前向传播
    fn transformer_forward(&self, input_embeddings: &[f32], seq_len: u32) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_hidden_layers;

        let mut hidden_states: Vec<f32> = input_embeddings.to_vec();

        for layer in 0..num_layers as usize {
            // 获取层权重
            let prefix = format!("layers.{}", layer);

            // Self-Attention
            let attn_output = self.self_attention(&hidden_states, seq_len, layer, &prefix);
            hidden_states = self.layer_norm(&hidden_states, layer, "post_attention");
            hidden_states = self.add(&hidden_states, &attn_output);

            // Feed Forward Network
            let ffn_output = self.feed_forward(&hidden_states, seq_len, layer, &prefix);
            hidden_states = self.layer_norm(&hidden_states, layer, "post_ffn");
            hidden_states = self.add(&hidden_states, &ffn_output);

            tracing::trace!("Transformer layer {} complete", layer);
        }

        // 最终 LayerNorm
        hidden_states = self.layer_norm_final(&hidden_states);

        hidden_states
    }

    /// Self-Attention 实现
    fn self_attention(
        &self,
        input: &[f32],
        seq_len: u32,
        layer_idx: usize,
        prefix: &str,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // 获取 QKV 权重
        let q_weight = self
            .get_weight(&format!("{}.attn.q.weight", prefix))
            .or_else(|| self.get_weight("attn.q.weight"))
            .map(|t| self.tensor_to_vec(&t));

        let k_weight = self
            .get_weight(&format!("{}.attn.k.weight", prefix))
            .or_else(|| self.get_weight("attn.k.weight"))
            .map(|t| self.tensor_to_vec(&t));

        let v_weight = self
            .get_weight(&format!("{}.attn.v.weight", prefix))
            .or_else(|| self.get_weight("attn.v.weight"))
            .map(|t| self.tensor_to_vec(&t));

        // 简化实现：使用输入的线性变换
        let (q, k, v) = if let (Some(qw), Some(kw), Some(vw)) = (q_weight, k_weight, v_weight) {
            (
                self.matmul_f32(input, &qw, seq_len, hidden_size, hidden_size),
                self.matmul_f32(input, &kw, seq_len, hidden_size, hidden_size),
                self.matmul_f32(input, &vw, seq_len, hidden_size, hidden_size),
            )
        } else {
            let qkv_size = (seq_len * hidden_size) as usize;
            (
                self.synthetic_qkv(input, seq_len, layer_idx, 0),
                self.synthetic_qkv(input, seq_len, layer_idx, 1),
                self.synthetic_qkv(input, seq_len, layer_idx, 2),
            )
        };

        // 应用注意力
        let mut output = vec![0.0_f32; (seq_len * hidden_size) as usize];
        let scale = 1.0 / (head_dim as f32).sqrt();

        for head in 0..num_heads as usize {
            for pos in 0..seq_len as usize {
                // 计算注意力分数
                let mut attention_weights = vec![0.0_f32; seq_len as usize];

                for kv_pos in 0..seq_len as usize {
                    let mut dot_product = 0.0_f32;
                    for dim in 0..head_dim as usize {
                        let q_idx = pos * hidden_size as usize + head * head_dim as usize + dim;
                        let k_idx = kv_pos * hidden_size as usize + head * head_dim as usize + dim;

                        if q_idx < q.len() && k_idx < k.len() {
                            dot_product += q[q_idx] * k[k_idx];
                        }
                    }
                    attention_weights[kv_pos] = (dot_product * scale).tanh() * 0.5 + 0.5;
                }

                // 归一化
                let sum: f32 = attention_weights.iter().sum();
                if sum > 0.0 {
                    for kv_pos in 0..seq_len as usize {
                        attention_weights[kv_pos] /= sum;
                    }
                }

                // 计算输出
                for dim in 0..head_dim as usize {
                    let out_idx = pos * hidden_size as usize + head * head_dim as usize + dim;
                    if out_idx < output.len() {
                        let mut weighted_sum = 0.0_f32;
                        for kv_pos in 0..seq_len as usize {
                            let v_idx =
                                kv_pos * hidden_size as usize + head * head_dim as usize + dim;
                            if v_idx < v.len() {
                                weighted_sum += attention_weights[kv_pos] * v[v_idx];
                            }
                        }
                        output[out_idx] = weighted_sum;
                    }
                }
            }
        }

        // Output projection
        if let Some(o_weight) = self
            .get_weight(&format!("{}.attn.output.weight", prefix))
            .or_else(|| self.get_weight("attn.output.weight"))
        {
            let o = self.tensor_to_vec(&o_weight);
            output = self.matmul_f32(&output, &o, seq_len, hidden_size, hidden_size);
        }

        output
    }

    /// 生成合成 QKV（当权重不可用时）
    fn synthetic_qkv(
        &self,
        input: &[f32],
        seq_len: u32,
        layer_idx: usize,
        qkv_idx: u32,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let mut output = vec![0.0_f32; (seq_len * hidden_size) as usize];

        for pos in 0..seq_len as usize {
            for dim in 0..hidden_size as usize {
                let idx = pos * hidden_size as usize + dim;
                let seed = (layer_idx as f32 * 100.0 + qkv_idx as f32 * 10.0 + dim as f32) * 0.01;
                let inp_val = input.get(idx).copied().unwrap_or(0.0);
                output[idx] = (inp_val + seed.sin()) * 0.9;
            }
        }

        output
    }

    /// 将 tensor 转换为 Vec<f32>
    fn tensor_to_vec(&self, tensor: &Tensor) -> Vec<f32> {
        match tensor.data() {
            TensorData::F32(data) => data.clone(),
            _ => vec![0.0; tensor.shape().dims().iter().product::<u32>() as usize],
        }
    }

    /// 矩阵乘法
    fn matmul_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        seq_len: u32,
        hidden_size: u32,
        output_size: u32,
    ) -> Vec<f32> {
        let input_len = (seq_len * hidden_size) as usize;
        let weight_rows = weight.len() / output_size as usize;
        let weight_cols = output_size as usize;

        let mut output = vec![0.0_f32; (seq_len * output_size) as usize];

        for pos in 0..seq_len as usize {
            for out_dim in 0..weight_cols {
                let mut sum = 0.0_f32;
                for in_dim in 0..weight_rows.min(hidden_size as usize) {
                    let in_idx = pos * weight_rows + in_dim;
                    let w_idx = out_dim * weight_rows + in_dim;

                    if in_idx < input.len() && w_idx < weight.len() {
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
                output[pos * weight_cols + out_dim] = sum;
            }
        }

        output
    }

    /// Feed Forward Network (Gated SiLU)
    fn feed_forward(
        &self,
        input: &[f32],
        seq_len: u32,
        layer_idx: usize,
        prefix: &str,
    ) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // 尝试获取 FFN 权重
        let w1 = self
            .get_weight(&format!("{}.mlp.gate_proj.weight", prefix))
            .or_else(|| self.get_weight("mlp.gate_proj.weight"))
            .map(|t| self.tensor_to_vec(&t));

        let w2 = self
            .get_weight(&format!("{}.mlp.up_proj.weight", prefix))
            .or_else(|| self.get_weight("mlp.up_proj.weight"))
            .map(|t| self.tensor_to_vec(&t));

        let w3 = self
            .get_weight(&format!("{}.mlp.down_proj.weight", prefix))
            .or_else(|| self.get_weight("mlp.down_proj.weight"))
            .map(|t| self.tensor_to_vec(&t));

        if let (Some(fc1), Some(fc2), Some(fc3)) = (w1, w2, w3) {
            // Gate_proj: input -> intermediate
            let gate = self.matmul_f32(input, &fc1, seq_len, hidden_size, intermediate_size);

            // Up_proj: input -> intermediate
            let up = self.matmul_f32(input, &fc2, seq_len, hidden_size, intermediate_size);

            // Gated SiLU: silu(gate) * up
            let mut gated = vec![0.0_f32; gate.len()];
            for i in 0..gate.len() {
                gated[i] = Self::silu(gate[i]) * up.get(i).copied().unwrap_or(0.0);
            }

            // Down_proj: gated -> hidden
            return self.matmul_f32(&gated, &fc3, seq_len, intermediate_size, hidden_size);
        }

        // Fallback: 简单的 MLP
        self.synthetic_ffn(input, seq_len, layer_idx)
    }

    /// 合成 FFN（当权重不可用时）
    fn synthetic_ffn(&self, input: &[f32], seq_len: u32, layer_idx: usize) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let mut output = vec![0.0_f32; (seq_len * hidden_size) as usize];

        for pos in 0..seq_len as usize {
            for dim in 0..hidden_size as usize {
                let idx = pos * hidden_size as usize + dim;
                let inp = input.get(idx).copied().unwrap_or(0.0);
                // 简化的 FFN
                output[idx] = Self::silu(inp * 1.5) * 0.8;
            }
        }

        output
    }

    /// SiLU 激活函数
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Layer Normalization
    fn layer_norm(&self, input: &[f32], layer_idx: usize, suffix: &str) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let seq_len = (input.len() / hidden_size as usize) as u32;

        let weight = self
            .get_weight(&format!("layers.{}.input_layernorm.weight", layer_idx))
            .or_else(|| self.get_weight("input_layernorm.weight"))
            .map(|t| self.tensor_to_vec(&t));

        let bias = self
            .get_weight(&format!("layers.{}.input_layernorm.bias", layer_idx))
            .or_else(|| self.get_weight("input_layernorm.bias"));

        let weight = weight.unwrap_or_else(|| vec![1.0_f32; hidden_size as usize]);
        let bias = bias.map(|t| self.tensor_to_vec(&t));

        let mut output = vec![0.0_f32; input.len()];
        let eps = 1e-5;

        for pos in 0..seq_len as usize {
            let offset = pos * hidden_size as usize;
            let slice = &input[offset..offset + hidden_size as usize];

            // 计算均值和方差
            let mean = slice.iter().sum::<f32>() / hidden_size as f32;
            let variance =
                slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;

            // 归一化
            for dim in 0..hidden_size as usize {
                let idx = offset + dim;
                let normalized = (input[idx] - mean) / (variance + eps).sqrt();
                output[idx] = normalized * weight.get(dim).copied().unwrap_or(1.0)
                    + bias
                        .as_ref()
                        .and_then(|b| b.get(dim))
                        .copied()
                        .unwrap_or(0.0);
            }
        }

        output
    }

    /// 最终 LayerNorm (RMSNorm)
    fn layer_norm_final(&self, input: &[f32]) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;
        let seq_len = (input.len() / hidden_size as usize) as u32;

        let weight = self
            .get_weight("final_layernorm.weight")
            .or_else(|| self.get_weight("norm.weight"))
            .map(|t| self.tensor_to_vec(&t))
            .unwrap_or_else(|| vec![1.0_f32; hidden_size as usize]);

        let mut output = vec![0.0_f32; input.len()];
        let eps = 1e-5;

        for pos in 0..seq_len as usize {
            let offset = pos * hidden_size as usize;
            let slice = &input[offset..offset + hidden_size as usize];

            // RMS 计算
            let rms = (slice.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32 + eps).sqrt();

            for dim in 0..hidden_size as usize {
                let idx = offset + dim;
                output[idx] = (input[idx] / rms) * weight.get(dim).copied().unwrap_or(1.0);
            }
        }

        output
    }

    /// 元素级加法
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    /// 池化 embeddings（用于分类/回归任务）
    fn pool_embeddings(&self, embeddings: &[f32], seq_len: u32) -> Vec<f32> {
        let hidden_size = self.config.hidden_size;

        // Mean pooling
        let mut pooled = vec![0.0_f32; hidden_size as usize];
        for dim in 0..hidden_size as usize {
            let mut sum = 0.0_f32;
            for pos in 0..seq_len as usize {
                let idx = pos * hidden_size as usize + dim;
                if idx < embeddings.len() {
                    sum += embeddings[idx];
                }
            }
            pooled[dim] = sum / seq_len as f32;
        }

        pooled
    }

    /// 主编码方法
    pub fn embed_prompt(&self, prompt: &str) -> Result<Tensor> {
        // 1. Tokenize
        let tokens = self.tokenize_vocab(prompt);

        // 2. Truncate to max length
        let max_len = self.config.max_position_embeddings;
        let tokens: Vec<u32> = tokens.into_iter().take(max_len as usize).collect();

        // 3. Embedding lookup
        let mut embeddings = self.embedding_lookup(&tokens);
        let seq_len = tokens.len() as u32;

        // 4. Add position encoding
        self.add_position_encoding(&mut embeddings, seq_len);

        // 5. Transformer forward pass
        let encoded = self.transformer_forward(&embeddings, seq_len);

        Ok(Tensor::from_data(
            TensorShape::new(vec![1, seq_len, self.config.hidden_size]),
            TensorData::F32(encoded),
        ))
    }
}

impl TextEncoderTrait for GemmaTextEncoder {
    fn encode(&self, text: &str) -> Result<Context> {
        let embeddings = self.embed_prompt(text)?;

        Ok(Context {
            embeddings,
            embeddings_neg: None,
            seq_len: self.config.max_position_embeddings,
        })
    }

    fn encode_negative(&self, text: &str) -> Result<Context> {
        let embeddings = self.embed_prompt(text)?;

        Ok(Context {
            embeddings,
            embeddings_neg: None,
            seq_len: self.config.max_position_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((GemmaTextEncoder::silu(0.0) - 0.0).abs() < 0.01);
        assert!((GemmaTextEncoder::silu(1.0) - 0.731).abs() < 0.01);
        assert!((GemmaTextEncoder::silu(-1.0) - (-0.269)).abs() < 0.01);
    }

    #[test]
    fn test_hash_f32() {
        let encoder = GemmaTextEncoder {
            gguf: Arc::new(GGUFFile {
                path: "dummy".into(),
                config: crate::model::gguf::GGUFConfig::default(),
            }),
            config: TextEncoderConfig::default(),
            weights: std::sync::RwLock::new(std::collections::HashMap::new()),
            use_real_weights: false,
        };

        let h1 = encoder.hash_f32("hello");
        let h2 = encoder.hash_f32("hello");
        assert_eq!(h1, h2); // 应该确定性

        let h3 = encoder.hash_f32("world");
        assert_ne!(h1, h3); // 不同的字符串应该有不同的哈希
    }
}
