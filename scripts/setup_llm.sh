#!/bin/bash

# Semantic Matcher - LLM Setup Script
# This script helps configure API keys for the novel class detection system

set -e

echo "=========================================="
echo "Semantic Matcher - LLM API Setup"
echo "=========================================="
echo ""

# Check if .env file exists
if [ -f .env ]; then
    echo "✓ .env file already exists"
    echo ""
    read -p "Do you want to update it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Existing .env file unchanged."
        exit 0
    fi
fi

# Copy .env.example if .env doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✓ Created .env from .env.example"
    else
        echo "✗ .env.example not found. Creating basic .env file..."
        cat > .env << 'EOF'
# Semantic Matcher - Environment Variables

# LLM Provider API Keys
OPENROUTER_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Default provider
LLM_CLASS_PROPOSER_PROVIDER=openrouter
LLM_CLASS_PROPOSER_MODEL=anthropic/claude-sonnet-4
EOF
    fi
fi

echo ""
echo "=========================================="
echo "Choose your LLM provider:"
echo "=========================================="
echo "1) OpenRouter (recommended - supports Claude, GPT-4, Gemini)"
echo "2) Anthropic (Claude models only)"
echo "3) OpenAI (GPT models only)"
echo "4) Skip (configure manually)"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "OpenRouter selected"
        echo "Get your API key at: https://openrouter.ai/keys"
        echo ""
        read -p "Enter your OpenRouter API key (sk-or-v1-...): " openrouter_key
        sed -i.bak "s|^OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=$openrouter_key|" .env
        rm .env.bak 2>/dev/null || true

        # Set as default provider
        sed -i.bak "s|^LLM_CLASS_PROPOSER_PROVIDER=.*|LLM_CLASS_PROPOSER_PROVIDER=openrouter|" .env
        sed -i.bak "s|^LLM_CLASS_PROPOSER_MODEL=.*|LLM_CLASS_PROPOSER_MODEL=anthropic/claude-sonnet-4|" .env
        rm .env.bak 2>/dev/null || true

        echo "✓ OpenRouter configured as default provider"
        ;;

    2)
        echo ""
        echo "Anthropic selected"
        echo "Get your API key at: https://console.anthropic.com/"
        echo ""
        read -p "Enter your Anthropic API key (sk-ant-...): " anthropic_key
        sed -i.bak "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$anthropic_key|" .env
        rm .env.bak 2>/dev/null || true

        # Set as default provider
        sed -i.bak "s|^LLM_CLASS_PROPOSER_PROVIDER=.*|LLM_CLASS_PROPOSER_PROVIDER=anthropic|" .env
        sed -i.bak "s|^LLM_CLASS_PROPOSER_MODEL=.*|LLM_CLASS_PROPOSER_MODEL=claude-sonnet-4|" .env
        rm .env.bak 2>/dev/null || true

        echo "✓ Anthropic configured as default provider"
        ;;

    3)
        echo ""
        echo "OpenAI selected"
        echo "Get your API key at: https://platform.openai.com/api-keys"
        echo ""
        read -p "Enter your OpenAI API key (sk-...): " openai_key
        sed -i.bak "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=$openai_key|" .env
        rm .env.bak 2>/dev/null || true

        # Set as default provider
        sed -i.bak "s|^LLM_CLASS_PROPOSER_PROVIDER=.*|LLM_CLASS_PROPOSER_PROVIDER=openai|" .env
        sed -i.bak "s|^LLM_CLASS_PROPOSER_MODEL=.*|LLM_CLASS_PROPOSER_MODEL=gpt-4o|" .env
        rm .env.bak 2>/dev/null || true

        echo "✓ OpenAI configured as default provider"
        ;;

    4)
        echo ""
        echo "Skipping. Configure your API keys manually in .env"
        ;;

    *)
        echo ""
        echo "Invalid choice. Configure your API keys manually in .env"
        ;;
esac

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test your configuration:"
echo "   uv run python -c 'from semanticmatcher.novelty import LLMClassProposer; print(\"LLM setup ready!\")'"
echo ""
echo "2. Run the example:"
echo "   uv run python examples/novel_discovery_example.py"
echo ""
echo "3. Check your API keys are working:"
echo "   uv run python -c "
echo "from semanticmatcher.novelty import LLMClassProposer; "
echo "proposer = LLMClassProposer(); "
echo "print(\"API keys configured:\", bool(proposer.api_keys))"
echo "'"
echo ""
