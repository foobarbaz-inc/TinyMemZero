import torch
from typing import Dict, Optional, Tuple, Union
from transformers import PreTrainedModel


class ModelWithToolUse(torch.nn.Module):
    def __init__(self, base_model: PreTrainedModel, tokenizer, memory_tool):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.memory_tool = memory_tool
        
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
        """Generation with tool use capability."""
        return self._generate_with_tool_use(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict:
        """Forward pass with tool handling during inference."""
        if self.training or labels is not None:
            # Regular training forward pass
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            # Inference mode with tool handling
            max_new_tokens = kwargs.pop('max_new_tokens', 50)
            return self._generate_with_tool_use(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

    def _generate_with_tool_use(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 50,
        **kwargs
    ) -> Dict:
        """Handle generation with tool interruption and continuation."""
        outputs = []
        tool_calls = []
        
        # Generate tokens one at a time
        current_ids = input_ids
        current_mask = attention_mask
        
        # Remove generation-specific kwargs that we don't want to pass to inner generate calls
        inner_kwargs = kwargs.copy()
        for key in ['return_dict_in_generate', 'output_scores', 'max_new_tokens']:
            inner_kwargs.pop(key, None)
        
        for _ in range(max_new_tokens):
            # Get next token prediction using generate() for one step
            outputs = self.base_model.generate(
                input_ids=current_ids,
                attention_mask=current_mask,
                max_new_tokens=1,
                **inner_kwargs
            )
            
            next_token_id = outputs[0, -1].unsqueeze(0)
            
            # Check for memory tool trigger
            if next_token_id.item() == self.memory_start_token_id:
                # Collect memory query and execute tool call
                query_ids, query_mask, memory_result = self._handle_memory_call(
                    current_ids,
                    current_mask,
                    inner_kwargs
                )
                
                # Record tool usage
                tool_calls.append({
                    "query": self.tokenizer.decode(query_ids[0]),
                    "result": memory_result
                })
                
                # Update current sequence with tool results
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
                # Regular token generation
                current_ids = torch.cat([
                    current_ids,
                    next_token_id.unsqueeze(0)
                ], dim=1)
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(1, 1).to(current_mask.device)
                ], dim=1)
            
            # Check for end of generation
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        return type('GenerationOutput', (), {
            'sequences': current_ids,
            'scores': None,  # Add if needed
            'tool_calls': tool_calls
        })

    def _handle_memory_call(
        self,
        current_ids: torch.Tensor,
        current_mask: torch.Tensor,
        generation_kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Handle memory tool call and return query tokens and result."""
        query_tokens = []
        
        while len(query_tokens) < 100:  # max query length safeguard
            outputs = self.base_model.generate(
                input_ids=current_ids,
                attention_mask=current_mask,
                max_new_tokens=1,
                **generation_kwargs
            )
            
            next_token_id = outputs[0, -1].item()
            query_tokens.append(next_token_id)
            
            if next_token_id == self.memory_end_token_id:
                break
            
            current_ids = torch.cat([
                current_ids,
                outputs[:, -1:],
            ], dim=1)
            current_mask = torch.cat([
                current_mask,
                torch.ones(1, 1).to(current_mask.device)
            ], dim=1)
        
        # Convert query tokens to text (excluding special tokens)
        query_text = self.tokenizer.decode(query_tokens[:-1])  # exclude end token
        
        # Execute memory tool call
        memory_result = self.memory_tool.query(query_text)
        
        return current_ids, current_mask, memory_result 