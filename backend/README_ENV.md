# Environment Variables Setup

## API Keys Required

The Cellian backend requires API keys for LLM functionality. These should be stored in a `.env` file.

### Required API Keys

1. **Google Gemini API Key** (Required for LLM features)
   - Variable names: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
   - Get your key from: https://makersuite.google.com/app/apikey
   - Used for: Query processing, perturbation extraction, hypothesis generation

2. **Edison Scientific API Key** (Optional, for literature search)
   - Variable name: `EDISON_API_KEY`
   - Get your key from: https://edison.scientific.ai/
   - Used for: Literature search and citation support in hypotheses
   - Note: System works without this, but literature search will be disabled

## Setup Instructions

1. **Copy the example file**:
   ```bash
   cd /home/nebius/cellian/backend
   cp .env.example .env
   ```

2. **Edit `.env` file** and add your actual API keys:
   ```bash
   nano .env
   # or
   vim .env
   ```

3. **Fill in your keys**:
   ```env
   GOOGLE_API_KEY=AIza...your_actual_key_here
   EDISON_API_KEY=your_edison_key_here
   ```

## .env File Locations

The backend will automatically load `.env` files from these locations (in order):
1. `backend/.env` (preferred location)
2. `backend/llm/.env` (fallback)
3. `cellian/.env` (project root, fallback)

The first `.env` file found will be used. You can place your `.env` file in any of these locations.

## Security

- ✅ `.env` files are already in `.gitignore` - they will NOT be committed to git
- ✅ Never commit API keys to version control
- ✅ Use `.env.example` as a template (without actual keys)
- ✅ Keep your `.env` file secure and don't share it

## Verification

After setting up your `.env` file, start the backend server:
```bash
cd /home/nebius/cellian/backend
conda activate new_env
python api.py
```

You should see:
```
✓ Loaded .env from /path/to/.env
✓ LLM modules loaded successfully
```

If you see warnings about missing API keys, check that:
1. Your `.env` file exists in one of the expected locations
2. The API key variable names are correct (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)
3. The keys are valid and not expired

