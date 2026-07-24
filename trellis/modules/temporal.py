import torch

def voxel_chunk(x: torch.Tensor, chunk_size: int):
    """
    chunk the voxel latent for sparse temproal attention.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        chunk_size (int): Chunk size
    """
    B, T, C, D, H, W = x.shape[:]
    num_chunks_per_dim = D // chunk_size # 16 / 4 = 4
    x = x.view(
        B, T, C,
        num_chunks_per_dim, chunk_size,  # D -> 4, 4
        num_chunks_per_dim, chunk_size,  # H -> 4, 4
        num_chunks_per_dim, chunk_size   # W -> 4, 4
    )

    x = x.permute(
        0, 3, 5, 7, # B, D_chunks, H_chunks, W_chunks 
        1, 2,      # T, C
        4, 6, 8     # D_size, H_size, W_size 
    )

    num_total_chunks = num_chunks_per_dim**3
    x = x.reshape(
        B, num_total_chunks, 
        T, C, 
        chunk_size, chunk_size, chunk_size
    )
    return x


def voxel_unchunk(x: torch.Tensor, chunk_size: int):
    """
    chunk the voxel latent for sparse temproal attention.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        chunk_size (int): Chunk size
    """
    B, num_total_chunks, T, C , _, _, _= x.shape[:]
    num_chunks_per_dim = round(num_total_chunks ** (1/3))
    D = H = W = int(num_chunks_per_dim * chunk_size)
    x = x.reshape(
        B, num_chunks_per_dim, num_chunks_per_dim, num_chunks_per_dim, # B, D_chunks, H_chunks, W_chunks
        T, C,                                                     # T, C
        chunk_size, chunk_size, chunk_size                           # D_size, H_size, W_size
    )

    x = x.permute(
        0, 4, 5,   # B, T, C
        1, 6,      # D_chunks, D_size
        2, 7,      # H_chunks, H_size
        3, 8       # W_chunks, W_size
    )

    x = x.reshape(
        B, T, C,
        D, H, W
    )
    return x