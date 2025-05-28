#!/usr/bin/env python3
import os
import json
import xml.etree.ElementTree as ET
from datasets import Dataset, DatasetDict
import random
import argparse
import re
import ast
from typing import List, Dict, Any, Optional, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define XML tags that will be used as special tokens
DEFAULT_SPECIAL_TOKENS = {
    # Basic structural tags
    "section_open": "<section>",
    "section_close": "</section>",
    "question_open": "<question>",
    "question_close": "</question>",
    "answer_open": "<answer>",
    "answer_close": "</answer>",
    "context_open": "<context>",
    "context_close": "</context>",
    "dialog_open": "<dialog>",
    "dialog_close": "</dialog>",
    # Additional tags for specialized datasets
    "instruction_open": "<instruction>",
    "instruction_close": "</instruction>",
    "document_open": "<document>",
    "document_close": "</document>",
    "text_open": "<text>",
    "text_close": "</text>",
    # Story and fiction tags
    "story_open": "<story>",
    "story_close": "</story>",
    "fiction_open": "<fiction>",
    "fiction_close": "</fiction>",
    # Title tags
    "title_open": "<title>",
    "title_close": "</title>",
    "think_open": "<think>",
    "think_close": "</think>",
    # Miscellaneous tags
    "note_open": "<note>",
    "note_close": "</note>",
    "warning_open": "<warning>",
    "warning_close": "</warning>",
    "error_open": "<error>",
    "error_close": "</error>",
    "hint_open": "<hint>",
    "hint_close": "</hint>",
    "document_title_open": "<document_title>",
    "document_title_close": "</document_title>",
    "paragraph_open": "<paragraph>",
    "paragraph_close": "</paragraph>",
    "list_open": "<list>",
    "list_close": "</list>",
    "item_open": "<item>",
    "item_close": "</item>",
    "link_open": "<link>",
    "link_close": "</link>",
    "code_open": "<code>",
    "code_close": "</code>",
    "quote_open": "<quote>",
    "quote_close": "</quote>",
    "example_open": "<example>",
    "example_close": "</example>",
    "question_id_open": "<question_id>",
    "question_id_close": "</question_id>",
    "document_id_open": "<document_id>",
    "document_id_close": "</document_id>",
    "passage_open": "<passage>",
    "passage_close": "</passage>",
    "metadata_open": "<metadata>",
    "metadata_close": "</metadata>",
    "ground_truth_open":"<ground_truth>",
    "ground_truth_close": "</ground_truth>",
    "rephrase_ground_truth_open": "<rephrase_ground_truth>",
    "rephrase_ground_truth_close": "</rephrase_ground_truth>",
    "explanation_open": "<explanation>",
    "explanation_close": "</explanation>",
}

# Mapping of common tag pairs that might contain instruction/input/output
TAG_MAPPING = {
    # Tag pairs that typically contain instructions
    "instruction": ["<instruction>", "</instruction>"],
    "question": ["<question>", "</question>"],
    
    # Tag pairs that typically contain input content
    "context": ["<context>", "</context>"],
    "document": ["<document>", "</document>"],
    "passage": ["<passage>", "</passage>"],
    
    # Tag pairs that typically contain expected output
    "answer": ["<answer>", "</answer>"],
    "ground_truth": ["<ground_truth>", "</ground_truth>"],
    "explanation": ["<explanation>", "</explanation>"],
}

# Define the target column for the output dataset
TARGET_COLUMN = "text"

def format_prompt(instruction, input_text=None):
    """
    Formats the instruction and input into a prompt string.
    """
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def create_training_text(instruction, input_text, output_text):
    """
    Combines the prompt and the expected output.
    """
    prompt = format_prompt(instruction, input_text)
    return f"{prompt}{output_text}"

def extract_content_by_tags(text, open_tag, close_tag):
    """
    Extract content between open and close tags from text.
    Returns a list of matched content.
    """
    pattern = f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def extract_instruction_input_output(text):
    """
    Extract instruction, input, and output from text using tags.
    Returns a tuple of (instruction, input, output).
    """
    instruction = None
    input_text = None
    output = None
    
    # Try to extract instruction
    for tag_type, (open_tag, close_tag) in TAG_MAPPING.items():
        if tag_type in ["instruction", "question"]:
            content = extract_content_by_tags(text, open_tag, close_tag)
            if content:
                instruction = content[0]
                break
    
    # Try to extract input
    for tag_type, (open_tag, close_tag) in TAG_MAPPING.items():
        if tag_type in ["context", "document", "passage"]:
            content = extract_content_by_tags(text, open_tag, close_tag)
            if content:
                input_text = content[0]
                break
    
    # Try to extract output
    for tag_type, (open_tag, close_tag) in TAG_MAPPING.items():
        if tag_type in ["answer", "ground_truth", "explanation"]:
            content = extract_content_by_tags(text, open_tag, close_tag)
            if content:
                output = content[0]
                break
    
    return instruction, input_text, output

def load_raw_text(file_path, file_specific_strategy=None):
    """
    Load raw text file with various strategies to extract data.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        
        # Strategy 1: Check if it's a python list-like format ["text1", "text2", ...]
        if content.strip().startswith('[') and content.strip().endswith(']'):
            try:
                # Try to parse as Python list
                items = ast.literal_eval(content)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            # Create a simple instruction for each item
                            instruction = "Complete the following text or respond appropriately:"
                            data.append({
                                TARGET_COLUMN: create_training_text(instruction, item, "")
                            })
                    return data
            except (SyntaxError, ValueError):
                pass  # Not a valid Python list, continue with other strategies
        
        # Strategy 2: Look for special tags in the text
        instruction, input_text, output = extract_instruction_input_output(content)
        
        if instruction and output:
            data.append({
                TARGET_COLUMN: create_training_text(instruction, input_text, output)
            })
            return data
            
        # Strategy 3: Split by newlines for line-by-line processing
        lines = content.split('\n')
        
        # Check if it's a Q&A format with alternating lines
        if len(lines) >= 2:
            for i in range(0, len(lines) - 1, 2):
                if lines[i].strip() and lines[i+1].strip():
                    instruction = lines[i].strip()
                    output = lines[i+1].strip()
                    data.append({
                        TARGET_COLUMN: create_training_text(instruction, None, output)
                    })
            
            if data:
                return data
                
        # Strategy 4: Fall back to text chunking for unsupported formats
        # This is a simplified approach - you might want to use a more sophisticated chunking strategy
        chunks = []
        current_chunk = ""
        
        for line in lines:
            current_chunk += line + "\n"
            if len(current_chunk) > 500:  # Arbitrary chunk size
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        for i in range(len(chunks) - 1):
            instruction = "Continue the following text:"
            input_text = chunks[i]
            output = chunks[i + 1]
            data.append({
                TARGET_COLUMN: create_training_text(instruction, input_text, output)
            })
            
    return data

def load_xml_data(file_path):
    """
    Load XML file with a flexible approach to handle different structures.
    """
    data = []
    try:
        # First, try conventional XML parsing
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Look for items or similar container elements
        for container in ['item', 'example', 'entry', 'record', 'data']:
            for item in root.findall(f'.//{container}'):
                instruction = None
                input_text = None
                output = None
                
                # Look for instruction in various tag names
                for instr_tag in ['instruction', 'question', 'task']:
                    instr_elem = item.find(f'.//{instr_tag}')
                    if instr_elem is not None and instr_elem.text:
                        instruction = instr_elem.text.strip()
                        break
                
                # Look for input in various tag names
                for input_tag in ['input', 'context', 'document', 'passage']:
                    input_elem = item.find(f'.//{input_tag}')
                    if input_elem is not None and input_elem.text:
                        input_text = input_elem.text.strip()
                        break
                
                # Look for output in various tag names
                for output_tag in ['output', 'answer', 'response', 'ground_truth', 'result']:
                    output_elem = item.find(f'.//{output_tag}')
                    if output_elem is not None and output_elem.text:
                        output = output_elem.text.strip()
                        break
                
                if instruction and output:
                    data.append({
                        TARGET_COLUMN: create_training_text(instruction, input_text, output)
                    })
        
        if data:
            return data
            
    except ET.ParseError:
        # If conventional XML parsing fails, try looking for XML-like tags in the text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
        # Extract using custom tags
        instruction, input_text, output = extract_instruction_input_output(content)
        
        if instruction and output:
            data.append({
                TARGET_COLUMN: create_training_text(instruction, input_text, output)
            })
    
    return data

def inspect_json_structure(json_data, max_depth=5, current_depth=0):
    """
    Recursively inspect JSON structure to understand its format.
    Returns a string description of the structure.
    """
    if current_depth >= max_depth:
        return "MAX_DEPTH_REACHED"
    
    if isinstance(json_data, dict):
        result = "{\n"
        for k, v in list(json_data.items())[:5]:  # Limit to first 5 keys for brevity
            v_repr = inspect_json_structure(v, max_depth, current_depth + 1)
            result += f"{' ' * (current_depth + 2)}{k}: {v_repr},\n"
        if len(json_data) > 5:
            result += f"{' ' * (current_depth + 2)}... ({len(json_data) - 5} more keys)\n"
        result += f"{' ' * current_depth}}}"
        return result
    elif isinstance(json_data, list):
        if not json_data:
            return "[]"
        if len(json_data) == 1:
            return f"[{inspect_json_structure(json_data[0], max_depth, current_depth + 1)}]"
        sample = json_data[0]
        sample_repr = inspect_json_structure(sample, max_depth, current_depth + 1)
        return f"[{sample_repr}, ... ({len(json_data) - 1} more items)]"
    elif isinstance(json_data, str):
        if len(json_data) > 50:
            return f'"{json_data[:50]}..."'
        return f'"{json_data}"'
    else:
        return str(json_data)

def load_json_data(file_path):
    """
    Load JSON file with a flexible approach to handle different structures.
    """
    data = []
    
    # Dictionary to store file-specific extraction functions
    file_specific_handlers = {
        "math_qa_pairs_clean.json": extract_math_qa_pairs,
        "science_qa_pairs_think_clean.json": extract_science_qa_pairs,
        "train_dialogue.json": extract_dialogue_dataset,
        "conversations.json": extract_conversations,
        "training_mix_data_2025_04_02.json": extract_training_mix,
        "training_mix_data_2025_04_04.json": extract_training_mix,
        # Add more file-specific handlers as needed
    }
    
    # For SD_ prefixed files (stories)
    if os.path.basename(file_path).startswith("SD_"):
        return extract_story_data(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            json_data = json.load(f)
            
            # Debug: Log the structure of the JSON
            filename = os.path.basename(file_path)
            logger.debug(f"Structure of {filename}:")
            structure_repr = inspect_json_structure(json_data)
            logger.debug(structure_repr)
            
            # Check if we have a specific handler for this file
            for pattern, handler in file_specific_handlers.items():
                if pattern in file_path:
                    logger.info(f"Using specialized handler for {pattern}")
                    return handler(json_data, file_path)
            
            # Case 1: List of objects with common patterns
            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict):
                        # Standard pattern
                        instruction = None
                        input_text = None
                        output = None
                        
                        # Look for instruction field with various possible names
                        for field in ['instruction', 'question', 'task', 'prompt', 'query', 'text']:
                            if field in item and item[field]:
                                instruction = item[field]
                                break
                        
                        # Look for input field with various possible names
                        for field in ['input', 'context', 'document', 'passage', 'content']:
                            if field in item and item[field]:
                                input_text = item[field]
                                break
                        
                        # Look for output field with various possible names
                        for field in ['output', 'answer', 'response', 'completion', 'ground_truth', 'target']:
                            if field in item and item[field]:
                                output = item[field]
                                break
                        
                        # Special case: If there's a "conversations" field, it might be a dialogue
                        if 'conversations' in item and isinstance(item['conversations'], list):
                            dialogue_data = extract_dialogue_from_conversations(item['conversations'])
                            if dialogue_data:
                                data.extend(dialogue_data)
                                continue
                        
                        # Only add if we have both instruction and output
                        if instruction and output:
                            data.append({
                                TARGET_COLUMN: create_training_text(instruction, input_text, output)
                            })
                            
                # Special case for Wikipedia-like entries
                if len(json_data) > 0 and not data:
                    # Check if this looks like Wikipedia content (just text entries)
                    all_strings = all(isinstance(item, str) for item in json_data)
                    if all_strings:
                        for text in json_data:
                            if len(text.strip()) > 100:  # Only use substantial entries
                                instruction = "Summarize the following text:"
                                data.append({
                                    TARGET_COLUMN: create_training_text(instruction, text, "")
                                })
            
            # Case 2: Object with a list of items under a field
            elif isinstance(json_data, dict):
                # Case 2.1: Check for common container fields
                for field in ['data', 'examples', 'items', 'dataset', 'records', 'conversations']:
                    if field in json_data and isinstance(json_data[field], list):
                        items = json_data[field]
                        if field == 'conversations' and isinstance(items, list):
                            dialogue_data = extract_dialogue_from_conversations(items)
                            if dialogue_data:
                                data.extend(dialogue_data)
                                continue
                                
                        for item in items:
                            if isinstance(item, dict):
                                instruction = None
                                input_text = None
                                output = None
                                
                                # Look for instruction field with various possible names
                                for subfield in ['instruction', 'question', 'task', 'prompt', 'query', 'text']:
                                    if subfield in item and item[subfield]:
                                        instruction = item[subfield]
                                        break
                                
                                # Look for input field with various possible names
                                for subfield in ['input', 'context', 'document', 'passage', 'content']:
                                    if subfield in item and item[subfield]:
                                        input_text = item[subfield]
                                        break
                                
                                # Look for output field with various possible names
                                for subfield in ['output', 'answer', 'response', 'completion', 'ground_truth', 'target']:
                                    if subfield in item and item[subfield]:
                                        output = item[subfield]
                                        break
                                
                                if instruction and output:
                                    data.append({
                                        TARGET_COLUMN: create_training_text(instruction, input_text, output)
                                    })
                        
                        if data:  # If we found data in this field, break out of loop
                            break
                
                # Case 2.2: Check for dialogue structure
                if 'messages' in json_data and isinstance(json_data['messages'], list):
                    dialogue_data = extract_dialogue_from_messages(json_data['messages'])
                    if dialogue_data:
                        data.extend(dialogue_data)
                
                # Case 2.3: Single example in JSON
                if not data:
                    instruction = None
                    input_text = None
                    output = None
                    
                    # Look for instruction field with various possible names
                    for field in ['instruction', 'question', 'task', 'prompt', 'query', 'text']:
                        if field in json_data and json_data[field]:
                            instruction = json_data[field]
                            break
                    
                    # Look for input field with various possible names
                    for field in ['input', 'context', 'document', 'passage', 'content']:
                        if field in json_data and json_data[field]:
                            input_text = json_data[field]
                            break
                    
                    # Look for output field with various possible names
                    for field in ['output', 'answer', 'response', 'completion', 'ground_truth', 'target']:
                        if field in json_data and json_data[field]:
                            output = json_data[field]
                            break
                    
                    if instruction and output:
                        data.append({
                            TARGET_COLUMN: create_training_text(instruction, input_text, output)
                        })
                        
                # Case 2.4: Story-like structure
                if not data and ('title' in json_data or 'content' in json_data):
                    title = json_data.get('title', '')
                    content = json_data.get('content', '')
                    if title and content:
                        instruction = f"Continue the story titled '{title}':"
                        # Split content into chunks for multiple examples
                        chunks = split_into_chunks(content, 1000)  # 1000 chars per chunk
                        for i in range(len(chunks) - 1):
                            data.append({
                                TARGET_COLUMN: create_training_text(
                                    instruction, 
                                    chunks[i], 
                                    chunks[i+1]
                                )
                            })
                
        # If no data was extracted by the normal methods, try extracting raw text
        if not data:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                # Check for embedded XML-like tags
                instruction, input_text, output = extract_instruction_input_output(content)
                if instruction and output:
                    data.append({
                        TARGET_COLUMN: create_training_text(instruction, input_text, output)
                    })
    
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from {file_path}: {str(e)}")
        # Try to extract as plain text
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                instruction, input_text, output = extract_instruction_input_output(content)
                if instruction and output:
                    data.append({
                        TARGET_COLUMN: create_training_text(instruction, input_text, output)
                    })
        except Exception as e:
            logger.error(f"Failed to extract as plain text: {str(e)}")
    
    if not data:
        logger.warning(f"No data extracted from {os.path.basename(file_path)}")
    else:
        logger.info(f"Extracted {len(data)} examples from {os.path.basename(file_path)}")
    
    return data

def extract_dialogue_from_conversations(conversations):
    """
    Extract instruction-response pairs from conversation-style data.
    """
    data = []
    
    if not conversations:
        return data
    
    # Track context to build history
    context = []
    
    for i, message in enumerate(conversations):
        # Skip if this is not a dict with required fields
        if not isinstance(message, dict):
            continue
            
        role = message.get('role', message.get('from', '')).lower()
        content = message.get('content', message.get('value', ''))
        
        if not content or not role:
            continue
            
        # If this is a user/human message and there's a next message
        if role in ['user', 'human', 'question'] and i < len(conversations) - 1:
            next_message = conversations[i+1]
            
            # Get the next message if it's from assistant/bot
            next_role = next_message.get('role', next_message.get('from', '')).lower()
            next_content = next_message.get('content', next_message.get('value', ''))
            
            if next_role in ['assistant', 'bot', 'ai', 'answer'] and next_content:
                # Current message is instruction, next message is the output
                if context:
                    # If we have context, include it as input
                    context_text = "\n".join(context)
                    data.append({
                        TARGET_COLUMN: create_training_text(content, context_text, next_content)
                    })
                else:
                    # No context, just instruction and output
                    data.append({
                        TARGET_COLUMN: create_training_text(content, None, next_content)
                    })
                    
                # Add this exchange to context
                context.append(f"User: {content}")
                context.append(f"Assistant: {next_content}")
                
                # Limit context to last 4 exchanges (8 messages)
                if len(context) > 8:
                    context = context[-8:]
    
    return data

def extract_dialogue_from_messages(messages):
    """
    Extract instruction-response pairs from messages array.
    """
    data = []
    
    if not messages:
        return data
    
    # Track context to build history
    context = []
    
    for i, message in enumerate(messages):
        # Skip if this is not a dict or string
        if not isinstance(message, (dict, str)):
            continue
            
        # Handle string messages (alternating roles assumed)
        if isinstance(message, str):
            content = message
            # Assume odd indices are user, even are assistant
            role = 'user' if i % 2 == 0 else 'assistant'
        else:
            # Handle dict messages
            role = message.get('role', message.get('from', message.get('speaker', ''))).lower()
            content = message.get('content', message.get('value', message.get('text', '')))
        
        if not content:
            continue
            
        # If this is a user/human message and there's a next message
        if role in ['user', 'human', 'question'] and i < len(messages) - 1:
            next_message = messages[i+1]
            
            # Get the next message details
            if isinstance(next_message, str):
                next_content = next_message
                next_role = 'assistant'  # Assume alternating roles
            else:
                next_role = next_message.get('role', next_message.get('from', next_message.get('speaker', ''))).lower()
                next_content = next_message.get('content', next_message.get('value', next_message.get('text', '')))
            
            if next_role in ['assistant', 'bot', 'ai', 'answer'] and next_content:
                # Current message is instruction, next message is the output
                if context:
                    # If we have context, include it as input
                    context_text = "\n".join(context)
                    data.append({
                        TARGET_COLUMN: create_training_text(content, context_text, next_content)
                    })
                else:
                    # No context, just instruction and output
                    data.append({
                        TARGET_COLUMN: create_training_text(content, None, next_content)
                    })
                    
                # Add this exchange to context
                context.append(f"User: {content}")
                context.append(f"Assistant: {next_content}")
                
                # Limit context to last 4 exchanges (8 messages)
                if len(context) > 8:
                    context = context[-8:]
    
    return data

def extract_math_qa_pairs(json_data, file_path):
    """
    Specialized extractor for math Q&A datasets.
    """
    data = []
    
    if isinstance(json_data, list):
        for item in json_data:
            if not isinstance(item, dict):
                continue
                
            question = item.get('question', '')
            answer = item.get('answer', '')
            explanation = item.get('explanation', '')
            
            if question and (answer or explanation):
                output = answer
                if explanation:
                    output += f"\n\nExplanation: {explanation}"
                
                data.append({
                    TARGET_COLUMN: create_training_text(question, None, output)
                })
    
    return data

def extract_science_qa_pairs(json_data, file_path):
    """
    Specialized extractor for science Q&A datasets with think tags.
    """
    data = []
    
    if isinstance(json_data, list):
        for item in json_data:
            if not isinstance(item, dict):
                continue
                
            question = item.get('question', '')
            answer = item.get('answer', '')
            thinking = item.get('thinking', '')
            
            if question and answer:
                output = answer
                if thinking:
                    # Format with thinking tags if present
                    output = f"{DEFAULT_SPECIAL_TOKENS['think_open']}\n{thinking}\n{DEFAULT_SPECIAL_TOKENS['think_close']}\n\n{answer}"
                
                data.append({
                    TARGET_COLUMN: create_training_text(question, None, output)
                })
    
    return data

def extract_dialogue_dataset(json_data, file_path):
    """
    Specialized extractor for dialogue datasets.
    """
    data = []
    
    if isinstance(json_data, dict) and 'data' in json_data:
        for item in json_data['data']:
            if not isinstance(item, dict):
                continue
                
            # Look for dialogue content
            if 'conversation' in item and isinstance(item['conversation'], list):
                dialogue_data = extract_dialogue_from_conversations(item['conversation'])
                data.extend(dialogue_data)
            elif 'messages' in item and isinstance(item['messages'], list):
                dialogue_data = extract_dialogue_from_messages(item['messages'])
                data.extend(dialogue_data)
    
    return data

def extract_conversations(json_data, file_path):
    """
    Specialized extractor for conversation datasets.
    """
    data = []
    
    if isinstance(json_data, list):
        for item in json_data:
            if not isinstance(item, dict):
                continue
                
            # Check for different conversation formats
            if 'messages' in item and isinstance(item['messages'], list):
                dialogue_data = extract_dialogue_from_messages(item['messages'])
                data.extend(dialogue_data)
            elif 'conversations' in item and isinstance(item['conversations'], list):
                dialogue_data = extract_dialogue_from_conversations(item['conversations'])
                data.extend(dialogue_data)
            elif 'message' in item and 'response' in item:
                # Simple message-response pair
                data.append({
                    TARGET_COLUMN: create_training_text(item['message'], None, item['response'])
                })
    
    return data

def extract_training_mix(json_data, file_path):
    """
    Specialized extractor for training mix datasets.
    """
    data = []
    
    if isinstance(json_data, list):
        for item in json_data:
            if not isinstance(item, dict):
                continue
                
            # Check various formats in training mix
            if 'instruction' in item and 'response' in item:
                instruction = item['instruction']
                input_text = item.get('input', None)
                output = item['response']
                
                data.append({
                    TARGET_COLUMN: create_training_text(instruction, input_text, output)
                })
            elif 'question' in item and 'answer' in item:
                question = item['question']
                context = item.get('context', None)
                answer = item['answer']
                
                data.append({
                    TARGET_COLUMN: create_training_text(question, context, answer)
                })
            elif 'messages' in item and isinstance(item['messages'], list):
                dialogue_data = extract_dialogue_from_messages(item['messages'])
                data.extend(dialogue_data)
    
    return data

def extract_story_data(file_path):
    """
    Specialized extractor for story datasets.
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            json_data = json.load(f)
            
            # Try to find story title and content
            title = None
            content = None
            
            if isinstance(json_data, dict):
                title = json_data.get('title', '')
                content = json_data.get('content', json_data.get('text', ''))
                
                # If no direct content, look in nested structures
                if not content and 'story' in json_data:
                    if isinstance(json_data['story'], str):
                        content = json_data['story']
                    elif isinstance(json_data['story'], dict):
                        content = json_data['story'].get('content', json_data['story'].get('text', ''))
            
            if content:
                # Create a title if none exists
                if not title:
                    title = os.path.basename(file_path).replace('.json', '').replace('SD_', '')
                
                instruction = f"Continue the story titled '{title}':"
                
                # Split content into chunks for multiple examples
                chunks = split_into_chunks(content, 1000)  # 1000 chars per chunk
                
                for i in range(len(chunks) - 1):
                    data.append({
                        TARGET_COLUMN: create_training_text(
                            instruction, 
                            chunks[i], 
                            chunks[i+1]
                        )
                    })
    
    except Exception as e:
        logger.error(f"Error processing story data from {file_path}: {str(e)}")
    
    return data

def split_into_chunks(text, chunk_size):
    """
    Split text into chunks of approximately chunk_size characters,
    trying to break at paragraph or sentence boundaries.
    """
    if not text or len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Try to find a paragraph break within the chunk size
        next_pos = current_pos + chunk_size
        if next_pos >= len(text):
            chunks.append(text[current_pos:])
            break
            
        # Look for paragraph breaks
        paragraph_break = text.rfind('\n\n', current_pos, next_pos)
        if paragraph_break != -1 and paragraph_break > current_pos + chunk_size // 2:
            # Found a good paragraph break
            chunks.append(text[current_pos:paragraph_break].strip())
            current_pos = paragraph_break + 2  # Skip the newlines
            continue
            
        # Look for single newline
        newline = text.rfind('\n', current_pos, next_pos)
        if newline != -1 and newline > current_pos + chunk_size // 2:
            # Found a newline
            chunks.append(text[current_pos:newline].strip())
            current_pos = newline + 1
            continue
            
        # Look for sentence breaks (period, question mark, exclamation mark)
        for punctuation in ['. ', '? ', '! ']:
            sentence_break = text.rfind(punctuation, current_pos, next_pos)
            if sentence_break != -1 and sentence_break > current_pos + chunk_size // 2:
                chunks.append(text[current_pos:sentence_break + 1].strip())
                current_pos = sentence_break + 2  # Skip the punctuation and space
                break
        else:  # No sentence break found
            # Just break at the chunk size
            chunks.append(text[current_pos:next_pos].strip())
            current_pos = next_pos
    
    return chunks

def load_jsonl_data(file_path):
    """
    Load JSONL file (one JSON object per line).
    """
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    instruction = None
                    input_text = None
                    output = None
                    
                    # Look for instruction field with various possible names
                    for field in ['instruction', 'question', 'task', 'prompt']:
                        if field in item and item[field]:
                            instruction = item[field]
                            break
                    
                    # Look for input field with various possible names
                    for field in ['input', 'context', 'document', 'passage']:
                        if field in item and item[field]:
                            input_text = item[field]
                            break
                    
                    # Look for output field with various possible names
                    for field in ['output', 'answer', 'response', 'completion', 'ground_truth']:
                        if field in item and item[field]:
                            output = item[field]
                            break
                    
                    if instruction and output:
                        data.append({
                            TARGET_COLUMN: create_training_text(instruction, input_text, output)
                        })
            except json.JSONDecodeError:
                continue
    
    return data

def save_jsonl(data, output_path):
    """
    Save data as JSONL for easy inspection.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for fine-tuning language models')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw datasets')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed datasets')
    parser.add_argument('--create_val_split', action='store_true', help='Create validation split')
    parser.add_argument('--val_split_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--save_jsonl', action='store_true', help='Save data in JSONL format for inspection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    all_data = []
    file_count = 0
    example_count = 0
    
    # Process all files in the input directory
    for root, _, files in os.walk(args.input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            if os.path.isfile(file_path):
                file_count += 1
                logger.info(f"Processing {file_path}...")
                
                try:
                    data = []
                    
                    # Process file based on extension
                    if filename.endswith(".txt"):
                        data = load_raw_text(file_path)
                    elif filename.endswith(".xml"):
                        data = load_xml_data(file_path)
                    elif filename.endswith(".json"):
                        data = load_json_data(file_path)
                    elif filename.endswith(".jsonl"):
                        data = load_jsonl_data(file_path)
                    else:
                        # Try to infer format from content
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            try:
                                content = f.read(1000)  # Read first 1000 chars to infer format
                                
                                if content.strip().startswith('{') or content.strip().startswith('['):
                                    # Looks like JSON
                                    data = load_json_data(file_path)
                                elif content.strip().startswith('<'):
                                    # Looks like XML
                                    data = load_xml_data(file_path)
                                else:
                                    # Try as raw text
                                    data = load_raw_text(file_path)
                            except UnicodeDecodeError:
                                logger.warning(f"Failed to read {file_path} as text. Skipping...")
                                continue
                    
                    if data:
                        all_data.extend(data)
                        example_count += len(data)
                        logger.info(f"Extracted {len(data)} examples from {filename}")
                    else:
                        logger.warning(f"No data extracted from {filename}")
                
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
    
    if not all_data:
        logger.error("No data processed. Exiting.")
        return
    
    logger.info(f"Processed {file_count} files")
    logger.info(f"Total examples extracted: {example_count}")
    
    # Shuffle data
    random.shuffle(all_data)
    
    # Create dataset splits
    if args.create_val_split:
        train_size = int((1 - args.val_split_ratio) * len(all_data))
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data)
        })
        
        # Save dataset
        dataset_dict.save_to_disk(os.path.join(args.output_dir, "tokenizable_dataset"))
        logger.info(f"Processed dataset saved to {args.output_dir}/tokenizable_dataset")
        logger.info(f"Train split: {len(train_data)} examples")
        logger.info(f"Validation split: {len(val_data)} examples")
        
        # Save JSONL if requested
        if args.save_jsonl:
            save_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
            save_jsonl(val_data, os.path.join(args.output_dir, "validation.jsonl"))
            logger.info(f"Saved JSONL files for inspection in {args.output_dir}")
    else:
        # Save as a single dataset
        dataset = Dataset.from_list(all_data)
        dataset.save_to_disk(os.path.join(args.output_dir, "tokenizable_dataset"))
        logger.info(f"Processed dataset saved to {args.output_dir}/tokenizable_dataset")
        
        # Save JSONL if requested
        if args.save_jsonl:
            save_jsonl(all_data, os.path.join(args.output_dir, "processed_data.jsonl"))
            logger.info(f"Saved JSONL file for inspection in {args.output_dir}")

if __name__ == "__main__":
    main()