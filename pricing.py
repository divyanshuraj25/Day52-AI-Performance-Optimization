"""USD per 1M tokens (input, output). Verified against public OpenAI price lists, Oct 2026 - re-check before relying on them."""
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "text-embedding-3-small": (0.02, 0.0)}
def cost_usd(model, tokens_in, tokens_out=0):
    pin, pout = PRICES[model]
    return (tokens_in * pin + tokens_out * pout) / 1_000_000
