from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, CLIPVisionModel

from lisaseg.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)

from lisaseg.model.llava.model.llava import LlavaLlamaForCausalLM
from lisaseg.model.segment_anything import build_sam_vit_h
from lisaseg.core.losses import dice_loss, sigmoid_ce_loss

class LISA(nn.Module):
    def __init__(
        self,
        local_rank,
        seg_token_idx,
        tokenizer,
        llm_version,
        lora_r,
        precision,
        load_in_4bit=False,
        load_in_8bit=False,
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        vision_tower="openai/clip-vit-large-patch14",
        mm_vision_select_layer=-2,
        freeze_lm=True,
        train_mask_decoder=True,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        vision_pretrained=None,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.tokenizer = tokenizer
        self.image_token = tokenizer.cls_token_id
        self.precision = precision
        self.ce_loss_weight = ce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.bce_loss_weight = bce_loss_weight

        # LLaVA
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        num_new_tokens = tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
        
        precision_configs = {
            "bf16": {
                "torch_dtype": torch.bfloat16,
                "load_in_4bit": False,
                "load_in_8bit": False,
                "quantization_config": None,
            },
            "fp16": {
                "torch_dtype": torch.half,
                "load_in_4bit": load_in_4bit,
                "load_in_8bit": load_in_8bit,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ) if load_in_4bit else None,
            },
            "default": {
                "torch_dtype": torch.float32,
                "load_in_4bit": False,
                "load_in_8bit": False,
                "quantization_config": None,
            }
        }

        precision_config = precision_configs.get(precision, precision_configs["default"])

        self.lm = LlavaLlamaForCausalLM.from_pretrained(
            llm_version,
            torch_dtype=precision_config["torch_dtype"],
            cache_dir=None,
            low_cpu_mem_usage=True,
            load_in_4bit=precision_config["load_in_4bit"],
            load_in_8bit=precision_config["load_in_8bit"],
            quantization_config=precision_config["quantization_config"],
            device_map="auto" if precision == "fp16" else None,
        )

        # Model Initialization
        self.lm.enable_input_require_grads()
        self.lm.gradient_checkpointing_enable()
        self.lm.config.use_cache = False
        self.lm.model.config.eos_token_id = tokenizer.eos_token_id
        self.lm.model.config.bos_token_id = tokenizer.bos_token_id
        self.lm.model.config.pad_token_id = tokenizer.pad_token_id

        # Initialize Vision Tower
        model_vision_dict = self.lm.get_model().initialize_vision_modules(
            vision_tower=vision_tower,
            mm_vision_select_layer=mm_vision_select_layer,
            precision=precision,
        )
        vision_config = model_vision_dict["vision_config"]
        vision_tower = self.lm.get_model().vision_tower[0]

        if vision_tower.device.type == "meta":
            vision_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.half,
            }.get(precision, torch.float32)
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower.config._name_or_path,
                torch_dtype=vision_dtype,
                low_cpu_mem_usage=True,
            ).cuda(local_rank)
            self.lm.get_model().vision_tower[0] = vision_tower
        else:
            vision_dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.half,
            }.get(precision, torch.float32)
            vision_tower.to(device="cuda", dtype=vision_dtype)

        # Configurations
        self.lm.config.tune_mm_mlp_adapter = False
        self.lm.config.freeze_mm_mlp_adapter = False
        self.lm.config.mm_use_im_start_end = True
        vision_config.use_im_start_end = True
        self.lm.config.sep_image_conv_front = False

        # Initialize Vision Tokenizer
        self.lm.initialize_vision_tokenizer(
            mm_use_im_start_end=True,
            tokenizer=tokenizer,
            num_new_tokens=num_new_tokens,
            device=local_rank,
            tune_mm_mlp_adapter=False,
        )

        # Freeze LM Parameters
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        # LoRA
        if lora_r > 0:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.lm = get_peft_model(self.lm, config)
            self.lm.print_trainable_parameters()

        # Additional Configurations and Setup
        self.llm_version = llm_version
        self.seg_token_idx = seg_token_idx
        self.lm.resize_token_embeddings(len(tokenizer))

        for n, p in self.lm.named_parameters():
            if any([x in n for x in ["lm_head", "embed_tokens"]]) and p.shape[0] == len(
                tokenizer
            ):
                p.requires_grad = True

        # SAM Setup
        self.visual_model = build_sam_vit_h(vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection Layer
        in_dim = self.lm.config.hidden_size
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings = self.visual_model.image_encoder(pixel_values)
        return image_embeddings

    def forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        # Compute image embeddings
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # Prepare segmentation token mask
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(self.local_rank)],
            dim=1,
        )

        if inference:
            # Inference mode
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            # Generate output hidden states
            output_hidden_states = []
            for i in range(1):  # Change to n_batch if needed
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = self.lm(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states = torch.cat(output_hidden_states, dim=0)
            output = None

        else:
            # Training mode
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            # Run LM model
            output = self.lm(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        # Compute hidden states
        hidden_states = self.text_hidden_fcs[0](output_hidden_states[-1])
        last_hidden_state = hidden_states.sum(dim=-1)

        # Prepare prediction embeddings
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]
        pred_embeddings_ = [pred_embeddings[seg_token_offset[i]:seg_token_offset[i+1]] for i in range(len(seg_token_offset) - 1)]

        # Generate prediction masks
        multimask_output = False
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings_):
            sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embedding.unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, iou_predictions = self.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss += mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.lm.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx

            hidden_states = [self.text_hidden_fcs[0](output_hidden_states)]
            last_hidden_state = hidden_states[0].sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_counts.cumsum(-1)], dim=0)
            seg_token_offset = seg_token_offset[seg_token_offset <= len(pred_embeddings)]
            pred_embeddings_ = [pred_embeddings[seg_token_offset[i]:seg_token_offset[i + 1]] for i in range(len(seg_token_offset) - 1)]

            image_embeddings = self.get_visual_embs(images)
            multimask_output = False
            pred_masks = []

            for i, pred_embedding in enumerate(pred_embeddings_):
                sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embedding.unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
                low_res_masks, _ = self.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                pred_mask = self.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
    # a297ba19c83e91e437fdaa7da52a11d22a0901c2