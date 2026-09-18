# Pull Request: NeMo Guardrails Integration with Cost Tracking

## Overview
This PR integrates **NeMo Guardrails** into the LLM orchestration pipeline, providing robust input and output content safety checks with cost and token usage tracking.
## Architecture

### Pipeline Flow
```
User Message
    ↓
[1] Input Guardrails Check ← NeMo + DSPy LLM
    ↓ (if allowed)
[2] Prompt Refinement ← DSPy
    ↓
[3] Chunk Retrieval ← Hybrid Retriever (Without Reranker)
    ↓
[4] Response Generation ← DSPy
    ↓
[5] Output Guardrails Check ← NeMo + DSPy LLM
    ↓ (if allowed)
Final Response + Complete Cost Breakdown
```

## How Guardrails Work

### 1. **Input Guardrails** (Before Processing)
**Purpose**: Validate user messages before expensive LLM operations

**Checks for**:
- Password/credential requests (self or others)
- Sensitive personal information (SSN, credit cards, private keys)
- Harmful, violent, or explicit content
- Jailbreak/prompt injection attempts
- Impersonation requests
- Rule circumvention attempts ("ignore instructions")
- Abusive/hateful language
- Malicious code or instructions
- System prompt extraction attempts
- Illegal activity requests

**Example Blocked Input**:
```
User: "What's my coworker's password?"
Guardrail: BLOCKED by InputRailException
Response: "I'm not able to respond to that request"
Cost: $0.000245 (10 tokens)
```

**Example Allowed Input**:
```
User: "How do I reset my own password?"
Guardrail: PASSED
Continues to prompt refinement
Cost: $0.000189 (8 tokens)
```

### 2. **Output Guardrails** (After Generation)
**Purpose**: Validate assistant responses before sending to user

**Checks for**:
- Leaked passwords/credentials
- Revealed sensitive information
- Harmful/violent/explicit content
- Abusive/offensive language
- Dangerous/illegal instructions
- Ethical violations
- Malicious code
- System prompt leakage

**Example Blocked Output**:
```
Generated: "John's password is abc123"
Guardrail: BLOCKED by OutputRailException
Response: "I cannot provide someone else's password"
Cost: $0.000312 (13 tokens)
```

**Example Allowed Output**:
```
Generated: "To reset your password, visit the portal..."
Guardrail: PASSED
Sent to user
Cost: $0.000156 (7 tokens)
```

## Technical Implementation

### Core Components

#### 1. **NeMoRailsAdapter** (`nemo_rails_adapter.py`)
- Manages guardrail lifecycle and initialization
- Implements `check_input()` and `check_output()` methods
- Tracks usage via `get_lm_usage_since()` utility
- Returns `GuardrailCheckResult` with cost data

**Key Features**:
- Lazy initialization (only creates Rails when first used)
- Native NeMo exception detection (when `enable_rails_exceptions: true`)
- Fallback pattern matching for reliability
- Conservative error handling (blocks on error)
- Comprehensive usage tracking

#### 2. **DSPyNeMoLLM** (`dspy_nemo_adapter.py`)
- Custom LangChain LLM provider for NeMo
- Bridges NeMo Guardrails ↔ DSPy LM
- Implements required LangChain interface:
  - `_call()` - Synchronous generation
  - `_acall()` - Async generation
  - `_generate()` - Batch processing
  - `_llm_type` - Provider identifier

**Design**:
- Uses `dspy.settings.lm` for actual generation
- Handles both string and list response formats
- Proper error propagation
- Async support via `asyncio.to_thread()`

#### 3. **GuardrailCheckResult** (Pydantic Model)
```python
class GuardrailCheckResult(BaseModel):
    allowed: bool  # True if content passes
    verdict: str  # "yes" = blocked, "no" = allowed
    content: str  # Response message
    blocked_by_rail: Optional[str]  # Exception type if blocked
    reason: Optional[str]  # Explanation
    error: Optional[str]  # Error message if failed
    usage: Dict[str, Union[float, int]]  # Cost tracking
```

### Detection Mechanisms

#### Primary: Exception Format (Reliable)
When `enable_rails_exceptions: true` in config:
```python
{
    "role": "exception",
    "content": {
        "type": "InputRailException",
        "message": "I'm not able to respond to that",
    },
}
```

#### Fallback: Pattern Matching (Safety Net)
If exception format unavailable:
- Checks for standard NeMo refusal phrases
- Logs warning to enable exception mode
- Still provides reliable blocking

### Cost Tracking Integration

**Similar to PromptRefiner**:
```python
# Record history before operation
history_length_before = len(lm.history) if lm else 0

# Perform guardrail check
result = adapter.check_input(user_message)

# Extract usage using centralized utility
usage_info = get_lm_usage_since(history_length_before)

# Store in result
result.usage = usage_info  # Contains: total_cost, tokens, num_calls
```

**Usage Dictionary Structure**:
```python
{
    "total_cost": 0.000245,  # USD
    "total_prompt_tokens": 8,
    "total_completion_tokens": 2,
    "total_tokens": 10,
    "num_calls": 1,
}
```

## Orchestration Integration

### Modified Pipeline in `llm_orchestration_service.py`

```python
costs_metric = {
    "input_guardrails": {...},  # Step 1
    "prompt_refiner": {...},  # Step 2
    "response_generator": {...},  # Step 4
    "output_guardrails": {...},  # Step 5
}

# Step 3 (retrieval) has no LLM cost
```

### Early Termination on Block

**Input Blocked**:
```python
if not input_result.allowed:
    return OrchestrationResponse(
        inputGuardFailed=True,
        content=input_result.content,  # Refusal message
    )
# Saves costs: no refinement, retrieval, or generation
```

**Output Blocked**:
```python
if not output_result.allowed:
    return OrchestrationResponse(
        content=output_result.content  # Safe alternative
    )
# Original response discarded
```

## Configuration

### Rails Config (`rails_config.py`)
```yaml
models:
  - type: main
    engine: dspy_custom      # Uses our DSPyNeMoLLM
    model: dspy-llm

enable_rails_exceptions: true  # CRITICAL for reliable detection

rails:
  input:
    flows:
      - self check input
  output:
    flows:
      - self check output

prompts:
  - task: self_check_input
    output_parser: is_content_safe
    content: |
      [Detailed safety policy with examples]
      
  - task: self_check_output
    output_parser: is_content_safe
    content: |
      [Detailed safety policy with examples]
```

## Cost Logging


```

LLM USAGE COSTS BREAKDOWN:

  input_guardrails    : $0.000245 (1 calls, 10 tokens)
  prompt_refiner      : $0.001234 (1 calls, 52 tokens)
  response_generator  : $0.004567 (1 calls, 189 tokens)
  output_guardrails   : $0.000312 (1 calls, 13 tokens)

  TOTAL               : $0.006358 (4 calls, 264 tokens)

```