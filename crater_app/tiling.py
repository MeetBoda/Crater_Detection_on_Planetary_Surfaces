import numpy as np
import cv2

TILE_SIZE = 512


def split_into_tiles(image, tile_size=TILE_SIZE):
    h, w = image.shape[:2]

    tiles = []
    positions = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]

            # pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size), dtype=image.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            tiles.append(tile)
            positions.append((x, y))

    return tiles, positions, (h, w)


def stitch_tiles(tile_outputs, positions, original_shape, tile_size=TILE_SIZE):
    h, w = original_shape

    full_mask = np.zeros((h, w), dtype=np.float32)

    for tile, (x, y) in zip(tile_outputs, positions):
        th, tw = tile.shape

        full_mask[y:y+th, x:x+tw] = tile[:h-y, :w-x]

    return full_mask