class DiPyGPUDevice:
    """Singleton GPU device for DIPY. Created once, reused forever."""
    _instance = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            cls._instance = adapter.request_device_sync()
        return cls._instance
    
    @classmethod  
    def available(cls):
        try:
            cls.get()
            return True
        except Exception:
            return False