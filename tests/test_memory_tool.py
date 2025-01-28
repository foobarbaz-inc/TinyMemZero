import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.models.model import ModelWithToolUse

class SimpleMemoryTool:
    """Simple memory tool that just prints to console and returns a fixed response."""
    def query(self, query_text: str) -> str:
        print("\n=== Memory Tool Called ===")
        print(f"Query: {query_text}")
        response = f"[Memory Result for: {query_text}]"
        print(f"Returning: {response}")
        print("========================\n")
        return response

def test_memory_tool_generation():
    # Initialize components
    model_name = "gpt2"  # Use a small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    memory_tool = SimpleMemoryTool()
    
    # Create model with tool use
    model = ModelWithToolUse(base_model, tokenizer, memory_tool)
    model.eval()  # Set to evaluation mode
    
    # Create a test prompt that should trigger memory use
    test_prompt = (
        "Let me think about this step by step:\n"
        "Please output <memory>query</memory> to look something up.\n"
        "What is Einstein's birth year?\n"
    )
    
    # Tokenize input
    inputs = tokenizer(
        test_prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    print("\n=== Starting Generation ===")
    print(f"Input prompt:\n{test_prompt}\n")
    
    # Generate with tool use
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Print results
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    print("\n=== Generation Results ===")
    print(f"Generated text:\n{generated_text}\n")

if __name__ == "__main__":
    test_memory_tool_generation() 