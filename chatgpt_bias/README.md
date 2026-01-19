# ChatGPT-Based Political Bias Detection

This module uses Azure OpenAI (ChatGPT) API calls to classify news articles as hyperpartisan or mainstream.

## Features

- **Prompt-based classification**: Uses carefully crafted system and user prompts to leverage ChatGPT's understanding of political bias
- **Cost control**: Configurable maximum articles per split via `MAX_ARTICLES_PER_SPLIT` in `config.py`
- **Caching**: Automatically caches API responses to avoid redundant calls and save costs
- **Rate limiting**: Built-in delays between API calls to respect rate limits

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Azure OpenAI credentials**:
   - Copy `.env.example` to `.env`
   - Fill in your Azure OpenAI credentials:
     ```
     AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
     AZURE_OPENAI_API_KEY=your-api-key-here
     AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
     AZURE_OPENAI_API_VERSION=2024-02-15-preview
     ```

3. **Adjust configuration** (optional):
   - Edit `chatgpt_bias/config.py`
   - Set `MAX_ARTICLES_PER_SPLIT` to control cost (default: 100)
   - Adjust `TEMPERATURE` for response determinism (default: 0.1)
   - Modify system/user prompts for different classification strategies

## Usage

### Standalone Evaluation

Evaluate ChatGPT model on test sets:

```bash
# Evaluate on both test splits (limited by MAX_ARTICLES_PER_SPLIT)
python -m chatgpt_bias.evaluate

# Evaluate on specific split
python -m chatgpt_bias.evaluate --split test_byarticle

# Override max articles limit
python -m chatgpt_bias.evaluate --max-articles 20

# Load previously saved results (no API calls)
python -m chatgpt_bias.evaluate --use-saved
```

**Note**: Once you run an evaluation, results are automatically saved to `cache/chatgpt_responses/results_*.json`. The root `evaluate.py` will automatically use these saved results to avoid re-running expensive API calls.

### Integrated Evaluation

The ChatGPT model is automatically included when running the root evaluation script:

```bash
python evaluate.py
```

If Azure OpenAI credentials are not configured, the model will be skipped with a warning.

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_ARTICLES_PER_SPLIT` | `100` | Maximum articles to evaluate per dataset split (set to `None` for no limit) |
| `TEMPERATURE` | `0.1` | Controls response randomness (lower = more deterministic) |
| `MAX_TOKENS` | `50` | Maximum tokens in API response |
| `SYSTEM_PROMPT` | _(see config)_ | Instructions for the AI model |
| `USER_PROMPT_TEMPLATE` | _(see config)_ | Template for article classification requests |

## How It Works

1. **Article Preprocessing**: Articles are truncated to 2000 characters to control API costs
2. **Prompt Construction**: System prompt defines the task, user prompt provides article content
3. **API Call**: ChatGPT analyzes article and responds with "HYPERPARTISAN" or "MAINSTREAM"
4. **Response Parsing**: Response is parsed to extract binary prediction (1 = hyperpartisan, 0 = mainstream)
5. **Caching**: Response is cached to disk to avoid redundant API calls
6. **Metrics Calculation**: Standard classification metrics computed from predictions

## Cache Location

API responses are cached in: `cache/chatgpt_responses/`

Each response is stored as a JSON file with structure:
```json
{
  "article_id": "0000001",
  "prediction": 1,
  "true_label": 1,
  "response_text": "HYPERPARTISAN",
  "title": "Article title...",
  "text_preview": "First 200 chars..."
}
```

## Cost Estimates

Based on typical GPT-3.5/GPT-4 pricing:

- **By-Article Test Set**: 628 articles
- **By-Publisher Test Set**: 750 articles
- **Total with default limit (100 per split)**: ~200 API calls
- **Estimated cost**: $0.20 - $2.00 depending on model and pricing tier

Always test with small batches first using `--max-articles 10`.

## Troubleshooting

**Error: Missing Azure OpenAI credentials**
- Ensure `.env` file exists with all required variables
- Verify credentials are valid by testing with a simple API call

**Error: Rate limit exceeded**
- Increase `rate_limit_delay` in `batch_classify()` call
- Reduce number of articles or split evaluation into smaller batches

**Unexpected predictions**
- Review cached responses in `cache/chatgpt_responses/`
- Adjust system/user prompts in `config.py`
- Increase `TEMPERATURE` for more varied responses
