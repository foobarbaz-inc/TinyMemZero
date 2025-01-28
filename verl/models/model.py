import torch
from typing import Dict, Optional, Tuple, Union
from transformers import PreTrainedModel


class ModelWithToolUse(torch.nn.Module):
    def __init__(self, base_model: PreTrainedModel, tokenizer, memory_tool):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.memory_tool = memory_tool
        self.max_length = 2048  # or whatever your model's max length is
        
        # Add special tokens if not present
        special_tokens = {
            "additional_special_tokens": ["<memory>", "</memory>"]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.base_model.resize_token_embeddings(len(tokenizer))

        # Cache token IDs for efficiency
        self.memory_start_token_id = self.tokenizer.convert_tokens_to_ids("<memory>")
        self.memory_end_token_id = self.tokenizer.convert_tokens_to_ids("</memory>")

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        **kwargs
    ) -> Dict:
        """Generation with tool use capability for inference."""
        batch_size = input_ids.size(0)
        all_sequences = []
        all_tool_calls = []
        
        # Remove generation-specific kwargs that we'll set explicitly
        generation_kwargs = kwargs.copy()
        for key in ['return_dict_in_generate', 'output_scores', 'max_new_tokens']:
            generation_kwargs.pop(key, None)
        
        for i in range(batch_size):
            current_ids = input_ids[i:i+1]
            current_mask = attention_mask[i:i+1]
            sequence_tool_calls = []
            
            for _ in range(max_new_tokens):
                # Generate next token
                outputs = self.base_model.generate(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generation_kwargs
                )
                next_token = outputs.sequences[:, -1:]
                
                # Check for memory tool trigger
                if next_token[0, 0].item() == self.memory_start_token_id:
                    # Collect memory query
                    query_ids = []
                    query_outputs = current_ids
                    
                    while True:
                        query_next = self.base_model.generate(
                            input_ids=query_outputs,
                            attention_mask=torch.ones_like(query_outputs),
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            **generation_kwargs
                        )
                        token = query_next.sequences[:, -1:]
                        query_ids.append(token)
                        query_outputs = torch.cat([query_outputs, token], dim=1)
                        
                        if token[0, 0].item() == self.memory_end_token_id:
                            break
                    
                    # Execute tool call
                    query_text = self.tokenizer.decode(torch.cat(query_ids, dim=1)[0])
                    memory_result = self.memory_tool.query(query_text)
                    sequence_tool_calls.append({"query": query_text, "result": memory_result})
                    
                    # Add result to sequence
                    result_ids = self.tokenizer.encode(
                        memory_result,
                        add_special_tokens=False,
                        return_tensors='pt'
                    ).to(current_ids.device)
                    
                    current_ids = torch.cat([current_ids, result_ids], dim=1)
                    current_mask = torch.cat([
                        current_mask,
                        torch.ones_like(result_ids)
                    ], dim=1)
                else:
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    current_mask = torch.cat([
                        current_mask,
                        torch.ones_like(next_token)
                    ], dim=1)
                
                if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                    break
            
            all_sequences.append(current_ids)
            all_tool_calls.append(sequence_tool_calls)
        
        # Pad sequences to same length
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            [seq[0] for seq in all_sequences],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        return type('GenerationOutput', (), {
            'sequences': padded_sequences,
            'tool_calls': all_tool_calls
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_tool_calls: bool = False,
        **kwargs
    ) -> Dict:
        """Forward pass with tool handling during training."""
        if not self.training and labels is None:
            # For inference without labels, use generate
            return self.generate(input_ids, attention_mask, **kwargs)
        
        batch_size = input_ids.size(0)
        tool_calls_per_sequence = [[] for _ in range(batch_size)]
        new_input_ids = []
        new_attention_masks = []
        new_labels = [] if labels is not None else None
        
        for i in range(batch_size):
            current_ids = input_ids[i:i+1]
            current_mask = attention_mask[i:i+1]
            current_labels = labels[i:i+1] if labels is not None else None
            
            # Process sequence until we hit a memory token or the end
            while current_ids.size(1) < self.max_length:
                outputs = self.base_model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    labels=current_labels,
                    **kwargs
                )
                
                next_token = torch.argmax(outputs.logits[:, -1:], dim=-1)
                
                if next_token.item() == self.memory_start_token_id:
                    # Handle memory tool call (similar to generate method)
                    query_ids = []
                    query_outputs = current_ids
                    
                    while True:
                        query_next = self.base_model.generate(
                            input_ids=query_outputs,
                            attention_mask=torch.ones_like(query_outputs),
                            max_new_tokens=1,
                            **kwargs
                        )
                        token = query_next[:, -1:]
                        query_ids.append(token)
                        query_outputs = torch.cat([query_outputs, token], dim=1)
                        
                        if token[0, 0].item() == self.memory_end_token_id:
                            break
                    
                    query_text = self.tokenizer.decode(torch.cat(query_ids, dim=1)[0])
                    memory_result = self.memory_tool.query(query_text)
                    tool_calls_per_sequence[i].append({
                        "query": query_text,
                        "result": memory_result
                    })
                    
                    result_ids = self.tokenizer.encode(
                        memory_result,
                        add_special_tokens=False,
                        return_tensors='pt'
                    ).to(current_ids.device)
                    
                    current_ids = torch.cat([current_ids, result_ids], dim=1)
                    current_mask = torch.cat([
                        current_mask,
                        torch.ones_like(result_ids)
                    ], dim=1)
                    
                    if current_labels is not None:
                        current_labels = torch.cat([
                            current_labels,
                            torch.full_like(result_ids, -100)
                        ], dim=1)
                else:
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    current_mask = torch.cat([
                        current_mask,
                        torch.ones_like(next_token)
                    ], dim=1)
                    
                    if current_labels is not None:
                        current_labels = torch.cat([
                            current_labels,
                            next_token
                        ], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            new_input_ids.append(current_ids)
            new_attention_masks.append(current_mask)
            if new_labels is not None:
                new_labels.append(current_labels)
        
        # Pad sequences
        padded_ids = torch.nn.utils.rnn.pad_sequence(
            [ids[0] for ids in new_input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        padded_masks = torch.nn.utils.rnn.pad_sequence(
            [mask[0] for mask in new_attention_masks],
            batch_first=True,
            padding_value=0
        )
        
        # Final forward pass
        outputs = self.base_model(
            input_ids=padded_ids,
            attention_mask=padded_masks,
            labels=torch.nn.utils.rnn.pad_sequence(
                [l[0] for l in new_labels],
                batch_first=True,
                padding_value=-100
            ) if new_labels is not None else None,
            **kwargs
        )
        
        if return_tool_calls:
            outputs['tool_calls'] = tool_calls_per_sequence
        
        return outputs 