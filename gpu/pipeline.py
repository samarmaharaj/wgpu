class PipelineCache:
    """Cache compiled shaders so they are not recompiled on every call."""
    _cache = {}
    
    @classmethod
    def get_or_compile(cls, shader_key, wgsl_source, constants, device):
        cache_key = (shader_key, tuple(sorted(constants.items())))
        if cache_key not in cls._cache:
            shader = device.create_shader_module(code=wgsl_source)
            pipeline = device.create_compute_pipeline(
                layout="auto",
                compute={"module": shader, "entry_point": "main",
                         "constants": constants}
            )
            cls._cache[cache_key] = pipeline
        return cls._cache[cache_key]