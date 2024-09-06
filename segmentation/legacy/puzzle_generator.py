import numpy as np
from PIL import Image
import random

class PuzzleGenerator:
    def __init__(self, difficulty='medium'):
        # Interface
        self.difficulty = difficulty
        self.puzzle_types = {
            'jigsaw': self.create_jigsaw_puzzle,
            'rotation': self.create_rotation_puzzle,
            'scramble': self.create_scramble_puzzle
        }

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty

    def generate_puzzle(self, image, puzzle_type='random'):
        if puzzle_type == 'random':
            puzzle_type = random.choice(list(self.puzzle_types.keys()))
        
        if puzzle_type not in self.puzzle_types:
            raise ValueError(f"Unknown puzzle type: {puzzle_type}")
        
        return self.puzzle_types[puzzle_type](image)

    def create_jigsaw_puzzle(self, image):
        tiles = self.get_difficulty_params()['jigsaw_tiles']
        return self._jigsaw_puzzle(image, tiles)

    def create_rotation_puzzle(self, image):
        max_rotations = self.get_difficulty_params()['max_rotations']
        return self._rotation_puzzle(image, max_rotations)

    def create_scramble_puzzle(self, image):
        scramble_factor = self.get_difficulty_params()['scramble_factor']
        return self._scramble_puzzle(image, scramble_factor)

    def _jigsaw_puzzle(self, image, tiles):
        width, height = image.size
        tile_width, tile_height = width // tiles, height // tiles
        
        # Create tiles
        puzzle_tiles = []
        for i in range(tiles):
            for j in range(tiles):
                box = (j * tile_width, i * tile_height, 
                       (j + 1) * tile_width, (i + 1) * tile_height)
                puzzle_tiles.append(image.crop(box))
        
        # Shuffle tiles
        random.shuffle(puzzle_tiles)
        
        # Reconstruct shuffled image
        shuffled_image = Image.new('RGB', (width, height))
        for i, tile in enumerate(puzzle_tiles):
            row = i // tiles
            col = i % tiles
            shuffled_image.paste(tile, (col * tile_width, row * tile_height))
        
        return shuffled_image, puzzle_tiles

    def _rotation_puzzle(self, image, max_rotations):
        width, height = image.size
        tiles = 3  # Fixed 3x3 grid for rotation puzzle
        tile_width, tile_height = width // tiles, height // tiles
        
        puzzle_tiles = []
        rotations = []
        for i in range(tiles):
            for j in range(tiles):
                box = (j * tile_width, i * tile_height, 
                       (j + 1) * tile_width, (i + 1) * tile_height)
                tile = image.crop(box)
                rotation = random.randint(0, max_rotations) * 90
                rotations.append(rotation)
                puzzle_tiles.append(tile.rotate(rotation))
        
        # Reconstruct rotated image
        rotated_image = Image.new('RGB', (width, height))
        for i, tile in enumerate(puzzle_tiles):
            row = i // tiles
            col = i % tiles
            rotated_image.paste(tile, (col * tile_width, row * tile_height))
        
        return rotated_image, rotations

    def _scramble_puzzle(self, image, scramble_factor):
        width, height = image.size
        pixels = np.array(image)
        
        # Scramble pixels
        num_scrambles = int(width * height * scramble_factor)
        for _ in range(num_scrambles):
            x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
            x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
            pixels[y1, x1], pixels[y2, x2] = pixels[y2, x2].copy(), pixels[y1, x1].copy()
        
        scrambled_image = Image.fromarray(pixels)
        return scrambled_image, num_scrambles

    def get_difficulty_params(self):
        params = {
            'easy': {'jigsaw_tiles': 2, 'max_rotations': 1, 'scramble_factor': 0.1},
            'medium': {'jigsaw_tiles': 3, 'max_rotations': 2, 'scramble_factor': 0.2},
            'hard': {'jigsaw_tiles': 4, 'max_rotations': 3, 'scramble_factor': 0.3}
        }
        return params[self.difficulty]

# Usage example
if __name__ == "__main__":
    # Load an image
    image_path = "path_to_your_math_notes_image.jpg"
    image = Image.open(image_path)

    # Create a puzzle generator
    generator = PuzzleGenerator(difficulty='medium')

    # Generate a random puzzle
    puzzle_image, puzzle_solution = generator.generate_puzzle(image)

    # Display or save the puzzle image
    puzzle_image.show()
    # puzzle_image.save("puzzle.jpg")

    print(f"Puzzle solution: {puzzle_solution}")