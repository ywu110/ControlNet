import torch

def get_time_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Get time embedding for the given timesteps.
    
    Args:
        timesteps (torch.Tensor): Timesteps to be embedded. of shape (batch_size,). 
        embedding_dim (int): Dimension of the embedding. 
        
    Returns:
        torch.Tensor: Time embeddings. of shape (batch_size, embedding_dim).
    """
    
    assert embedding_dim % 2 == 0, "Embedding dimension must be even."
    
    # half for cos, half for sin
    half_dim = embedding_dim // 2 
    
    # factor = 10000^(2i/d) = 10000^(i/half_dim)
    factor = 10000 ** (torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) / half_dim)
    
    # convert the timesteps from shape (batch_size,) to (batch_size, 1) to (batch_size, half_dim) to match the shape of factor
    time_emb = timesteps.unsqueeze(1).expand(-1, half_dim)
    # shape (batch_size, embedding_dim)
    time_emb = torch.cat([torch.sin(time_emb / factor), torch.cos(time_emb / factor)], dim=-1) 
        
    return time_emb