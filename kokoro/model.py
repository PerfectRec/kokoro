from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, List, Optional, Tuple, Union
import json
import torch
from torch.nn.utils.rnn import pad_sequence

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

    @torch.no_grad()
    def forward_batch(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speeds: torch.FloatTensor
    ) -> Tuple[List[torch.FloatTensor], List[torch.LongTensor]]:
        """
        Batched forward pass for multiple inputs using true batching.
        Optimized for GPU - all operations fully parallelized.
        
        Args:
            input_ids: Padded batch of input token IDs [batch_size, max_length]
            input_lengths: Actual length of each sequence [batch_size]
            ref_s: Voice embeddings for each item [batch_size, style_dim]
            speeds: Speed factors for each item [batch_size]
        
        Returns:
            Tuple of (audio_list, pred_dur_list) where each is a list of tensors
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Ensure input_lengths and speeds are on the correct device
        input_lengths = input_lengths.to(device)
        speeds = speeds.to(device)
        
        # Create text masks
        text_mask = torch.arange(input_lengths.max(), device=device).unsqueeze(0).expand(batch_size, -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)
        
        # BERT encoding
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        
        # Style processing
        s = ref_s[:, 128:]
        
        # Prosody prediction
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        
        # Duration prediction with speed adjustment
        attention_mask = (~text_mask).float()
        duration = self.predictor.duration_proj(x)
        pred_dur = torch.round(
            ((torch.sigmoid(duration)).sum(dim=-1).clamp(min=1) * attention_mask) / speeds.unsqueeze(1)
        ).long()
        
        # Calculate sequence lengths after duration expansion
        seq_lengths = pred_dur.sum(axis=-1)
        max_frames = seq_lengths.max().item()
        
        # Create batched alignment matrices using the kokoro_batch approach
        # This is the key to true batching - creates all alignments at once
        frame_indices = torch.arange(max_frames, device=device).view(1, 1, -1)  # [1, 1, max_frames]
        duration_cumsum = pred_dur.cumsum(dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Create masks for alignment
        mask1 = duration_cumsum > frame_indices  # [batch, seq_len, max_frames]
        mask2 = frame_indices >= torch.cat(
            [torch.zeros(batch_size, 1, 1, device=device), duration_cumsum[:, :-1, :]],
            dim=1
        )  # [batch, seq_len, max_frames]
        
        pred_aln_trgs = (mask1 & mask2).float()  # [batch, seq_len, max_frames]
        
        # Expand features to frame level using batched alignment - ALL AT ONCE!
        en = d.transpose(-1, -2) @ pred_aln_trgs  # [batch, channels, max_frames]
        
        # Text encoding - batched
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trgs  # [batch, channels, max_frames]
        
        # Create frame mask for padded positions
        frame_mask = (frame_indices.squeeze(1).expand(batch_size, -1) >= seq_lengths.unsqueeze(1)).to(device)
        frame_mask = (~frame_mask).float()  # [batch, max_frames]
        
        # F0 and N prediction - now batched with masking for efficiency!
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s, seq_lengths, frame_mask)
        
        # Decode audio - batched
        audio_batch = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        
        # Calculate audio lengths for all items at once (vectorized, no GPU sync)
        # The decoder upsamples by a factor, need to calculate actual audio samples
        audio_lengths = (seq_lengths * (audio_batch.shape[-1] / max_frames)).long()
        
        # Move everything to CPU in one go after GPU work is complete
        audio_batch_cpu = audio_batch.cpu()
        audio_lengths_cpu = audio_lengths.cpu()
        pred_dur_cpu = pred_dur.cpu()
        input_lengths_cpu = input_lengths.cpu()
        
        # Now extract individual audios (this is fast, all on CPU)
        audio_list = []
        pred_dur_list = []
        
        for i in range(batch_size):
            audio_len = audio_lengths_cpu[i].item()
            
            # Extract audio for this item (trim padding)
            if batch_size == 1:
                item_audio = audio_batch_cpu[:audio_len]
            else:
                item_audio = audio_batch_cpu[i, :audio_len]
            
            audio_list.append(item_audio)
            pred_dur_list.append(pred_dur_cpu[i, :input_lengths_cpu[i]])
        
        return audio_list, pred_dur_list

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
