from typing import List, Dict, Optional, Tuple
import torch
from transformers import PreTrainedTokenizer

class ToolHandler:
    def __init__(self, tokenizer: PreTrainedTokenizer, memory_tool):
        self.tokenizer = tokenizer
        self.memory_tool = memory_tool
        # Special tokens for memory tool use
        self.MEMORY_START_TOKEN = "<memory>"
        self.MEMORY_END_TOKEN = "</memory>"
        
        # Add special tokens to tokenizer if not already present
        special_tokens = {
            "additional_special_tokens": [self.MEMORY_START_TOKEN, self.MEMORY_END_TOKEN]
        }
        self.tokenizer.add_special_tokens(special_tokens)

    def handle_generation(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Handle generation with tool interruption and continuation.
        
        Returns:
            Tuple containing:
                - Generated token ids
                - List of tool calls made during generation
        """
        generated_ids = input_ids
        current_length = input_ids.shape[1]
        tool_calls = []
        
        while current_length < max_length:
            # Generate next token
            next_token_logits = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
            ).logits[:, -1, :]
            
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Check if it's a tool token
            if self._is_memory_start_token(next_token):
                # Collect the memory query
                query_tokens, query_text = self._collect_until_end_token(
                    model,
                    generated_ids,
                    attention_mask,
                    current_length + 1,  # Skip the start token
                    max_length,
                )
                
                # Execute memory tool call
                memory_result = self.memory_tool.query(query_text)
                tool_calls.append({
                    "query": query_text,
                    "result": memory_result
                })
                
                # Convert memory result to tokens and append
                result_tokens = self._tokenize_memory_result(memory_result)
                generated_ids = torch.cat([generated_ids, result_tokens], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones_like(result_tokens),
                ], dim=-1)
                
                current_length = generated_ids.shape[1]
            else:
                # Append regular token
                generated_ids = torch.cat([
                    generated_ids,
                    next_token.unsqueeze(0),
                ], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones_like(next_token.unsqueeze(0)),
                ], dim=-1)
                current_length += 1
        
        return generated_ids, tool_calls

    def _is_memory_start_token(self, token_id: int) -> bool:
        """Check if token is memory start token."""
        return token_id == self.tokenizer.convert_tokens_to_ids(self.MEMORY_START_TOKEN)

    def _collect_until_end_token(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_idx: int,
        max_length: int,
    ) -> Tuple[torch.Tensor, str]:
        """Collect tokens until memory end token is generated."""
        collected_tokens = []
        current_length = start_idx
        
        while current_length < max_length:
            next_token_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, -1, :]
            
            next_token = torch.argmax(next_token_logits, dim=-1)
            collected_tokens.append(next_token.item())
            
            if next_token == self.tokenizer.convert_tokens_to_ids(self.MEMORY_END_TOKEN):
                break
                
            current_length += 1
        
        # Convert collected tokens to text
        query_text = self.tokenizer.decode(collected_tokens[:-1])  # Exclude end token
        return torch.tensor(collected_tokens), query_text

    def _tokenize_memory_result(self, result: str) -> torch.Tensor:
        """Convert memory result to tokens."""
        return torch.tensor(self.tokenizer.encode(result, add_special_tokens=False))