from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, CLIPVisionModel

from lisaseg.model.llava.model.llava import LlavaLlamaForCausalLM
from lisaseg.model.segment_anything import build_sam_vit_h
from lisaseg.core.losses import dice_loss, sigmoid_ce_loss
from lisaseg.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN

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
        self.llm_version = llm_version
        self.seg_token_idx = seg_token_idx

        num_new_tokens = self.add_token_in_tokenizer(self.tokenizer)
        # Initialize LM Model
        self.lm = self.initialize_language_model(precision, llm_version, load_in_4bit, load_in_8bit)
        self.lm.enable_input_require_grads()
        self.lm.gradient_checkpointing_enable()
        self.set_lm_config(self.lm, tokenizer)
        self.set_lm_vision_tower(precision, self.lm, vision_tower, mm_vision_select_layer, local_rank)
        self.set_lm_vision_tokenizer(self.lm, tokenizer, num_new_tokens, local_rank)
        
        if freeze_lm:
            self.set_requires_grad(self.lm, False)
        # Set Peft LM
        self.lm = self.get_peft_lm(self.lm, lora_r, lora_alpha, lora_target_modules, lora_dropout)
        self.lm.resize_token_embeddings(len(tokenizer))
        self.set_condition_requires_grad(self.lm, True, ["lm_head", "embed_tokens"], tokenizer)

        # Set Visual SAM Model
        self.visual_model = self.initialize_vision_model(vision_pretrained)
        self.set_requires_grad(self.visual_model, False)
        if train_mask_decoder:
            self.visual_model.mask_decoder.train()
            self.set_requires_grad(self.visual_model.mask_decoder, True)
        # Set Projection Layer
        self.text_hidden_fcs = self.initialize_vision_projection(self.lm.config.hidden_size, out_dim)

    def add_token_in_tokenizer(self, tokenizer):
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        num_new_tokens = tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
        return num_new_tokens

    def initialize_language_model(self, precision, llm_version, load_in_4bit, load_in_8bit):
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
        lm = LlavaLlamaForCausalLM.from_pretrained(
            llm_version,
            torch_dtype=precision_config["torch_dtype"],
            cache_dir=None,
            low_cpu_mem_usage=True,
            load_in_4bit=precision_config["load_in_4bit"],
            load_in_8bit=precision_config["load_in_8bit"],
            quantization_config=precision_config["quantization_config"],
            device_map="auto" if precision == "fp16" else None,
        )
        return lm

    def set_lm_config(self, lm, tokenizer):
        lm.config.use_cache = False
        lm.model.config.eos_token_id = tokenizer.eos_token_id
        lm.model.config.bos_token_id = tokenizer.bos_token_id
        lm.model.config.pad_token_id = tokenizer.pad_token_id

        lm.config.tune_mm_mlp_adapter = False
        lm.config.freeze_mm_mlp_adapter = False
        lm.config.mm_use_im_start_end = True
        lm.config.sep_image_conv_front = False
        
    def set_lm_vision_tower(self, precision, lm, vision_tower, mm_vision_select_layer, local_rank):
        model_vision_dict = lm.get_model().initialize_vision_modules(
            vision_tower=vision_tower,
            mm_vision_select_layer=mm_vision_select_layer,
            precision=precision,
        )
        vision_config = model_vision_dict["vision_config"]
        vision_tower = lm.get_model().vision_tower[0]
        
        precision_dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.half,
            "fp32": torch.float32
        }
        if vision_tower.device.type == "meta":
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower.config._name_or_path,
                torch_dtype=precision_dtype_map.get(precision, torch.float32),
                low_cpu_mem_usage=True
            ).cuda(local_rank)
            lm.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device="cuda", dtype=precision_dtype_map.get(precision, torch.float32))
        vision_config.use_im_start_end = True
        
    def set_lm_vision_tokenizer(self, lm, tokenizer, num_new_tokens, local_rank):
        lm.initialize_vision_tokenizer(
            mm_use_im_start_end=True,
            tokenizer=tokenizer,
            num_new_tokens=num_new_tokens,
            device=local_rank,
            tune_mm_mlp_adapter=False,
        )
    
    def set_requires_grad(self, model, requires_grad):
        for n, param in model.named_parameters():
            param.requires_grad = requires_grad
            
    def set_condition_requires_grad(self, model, requires_grad, keys, tokenizer):
        for n, p in model.named_parameters():
            if any([x in n for x in keys]) and p.shape[0] == len(tokenizer):
                p.requires_grad = requires_grad
    
    def get_peft_lm(self, lm, lora_r, lora_alpha, lora_target_modules, lora_dropout):
        if lora_r > 0:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            lm = get_peft_model(lm, config)
            print("lora of peft for lm model.")
            lm.print_trainable_parameters()
            return lm
        return lm

    def initialize_vision_model(self, vision_pretrained):
        return build_sam_vit_h(vision_pretrained)

    def initialize_vision_projection(self, in_dim, out_dim):
        # Projection layer
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        return nn.ModuleList([nn.Sequential(*text_fc)])

    def get_vision_embs(self, pixel_values: torch.FloatTensor):
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
        image_embeddings = self.get_vision_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(self.local_rank),
            ],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = self.lm(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
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

            output = self.lm(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.text_hidden_fcs) == 1
        hidden_states.append(self.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
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

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = ce_loss
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

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

            hidden_states = []

            assert len(self.text_hidden_fcs) == 1
            hidden_states.append(self.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                sparse_embeddings, dense_embeddings = self.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
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
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
    
    