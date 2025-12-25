import torch


def average_heads(
    attention_map=None,
    attention_grad=None,
):
    """Copied from Chefer et al., (2021) https://github.com/hila-chefer/Transformer-Explainability. The function is used to average the attention maps from the different heads of the self-attention layer, weighted by gradients.

    Args:
        attention_map (_type_): _description_
        attention_grad (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Reshape attention map and attention grad to be of shape (batch_size*num_heads, seq_len, seq_len)
    attention_map = attention_map.reshape(
        -1, attention_map.shape[-2], attention_map.shape[-1]
    )

    if attention_grad.dim() == 4:
        attention_grad = attention_grad.reshape(
            -1, attention_grad.shape[-2], attention_grad.shape[-1]
        )
    weighted_attention_map = attention_grad * attention_map
    averaged_attention_map = weighted_attention_map.clamp(min=0).mean(dim=0)
    return averaged_attention_map


def apply_self_attention_rules(
    self_attn_R=None,
    attention_map=None,
):
    """Matrix multiplication to track the interactions between raw input features."""
    self_attn_R_addition = torch.matmul(attention_map, self_attn_R)
    return self_attn_R_addition


class InterpretationGenerator:
    def __init__(
        self,
        model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model
        self.model.zero_grad()
        self.model.eval()
        self.device = device

    def __call__(
        self,
        audio_inputs=None,
        audio_attn_masks=None,
        positive_class=True,
    ):
        audio_inputs = audio_inputs.to(self.device)
        audio_attn_masks = audio_attn_masks.to(self.device)

        output = self.model(
            audio_inputs=audio_inputs,
            audio_attn_masks=audio_attn_masks,
            get_attention_weights=True,
            average_attn_weights=False,
            register_hook=True,
        ).to(self.device)

        if positive_class:
            pos_class_output = output[:, 1]
            pos_class_output.backward(retain_graph=False)

        else:
            neg_class_output = output[:, 0]
            neg_class_output.backward(retain_graph=False)

    def handle_AST_self_attention(self, encoder_layers, sentence_index):
        for i, layer in enumerate(encoder_layers):
            grad = layer.attention.attention.attn_grad[0].detach().to("cpu")
            sentence_grad = grad[sentence_index]

            attn_map = layer.attention.attention.attn_map[0].detach().to("cpu")

            sentence_attn_map = attn_map[sentence_index]

            averaged_attn_map = average_heads(
                attention_map=sentence_attn_map,
                attention_grad=sentence_grad,
            )

            self.R_aa += apply_self_attention_rules(
                self_attn_R=self.R_aa,
                attention_map=averaged_attn_map,
            )

    def generate_sentence_level_explanations(
        self,
        sentence_index,
    ):
        # TODO Add summary of the function
        sentence_level_layers = (
            self.model.sentence_level_block.module.encoder.encoder.layer
        )

        num_spec_patches = (
            sentence_level_layers[0].attention.attention.attn_map[0].shape[-1]
        )

        # Initialize R_kk, which is the relevancy matrix for the audio modality
        self.R_aa = torch.eye(
            num_spec_patches,
            num_spec_patches,
        ).to("cpu")

        self.handle_AST_self_attention(sentence_level_layers, sentence_index)

        # Remove the first two tokens from audio modality because they are [cls] and
        # distillation token.
        spec_explanation = self.R_aa[0, 2:]

        return spec_explanation

    def generate_speech_level_interpretation(self):
        """Generates explanation for a speech-level data, indicating which sentence is important.

        Returns:
            _type_: return rows of R_ss, corresponding to the speech level [cls]
            token, which indicates the importance of sentence representations.
        """
        speech_level_layers = self.model.speech_level_block.layers

        num_sentences = speech_level_layers[0].self_attn.attn_map[0].shape[-2]

        self.R_ss = torch.eye(
            num_sentences,
            num_sentences,
        ).to("cpu")

        for i, layer in enumerate(speech_level_layers):
            grad = layer.self_attn.attn_grad[0].detach().to("cpu")
            attn_map = layer.self_attn.attn_map[0].detach().to("cpu")

            averaged_attn_map = average_heads(
                attention_map=attn_map,
                attention_grad=grad,
            )

            self.R_ss += apply_self_attention_rules(
                self_attn_R=self.R_ss,
                attention_map=averaged_attn_map,
            )

            relevancy = self.R_ss[0, 1:]
            min_val = relevancy.min()
            max_val = relevancy.max()

            normalized_relevancy = (relevancy - min_val) / (max_val - min_val)

        return normalized_relevancy
