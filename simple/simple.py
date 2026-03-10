from pathlib import Path
from wgpu_shadertoy import Shadertoy

shader_code = (Path(__file__).parent / "simple.wsgl").read_text()

shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()