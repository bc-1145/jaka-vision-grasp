"""Release SDK wrappers without issuing Stop twice.

Official wrappers terminate by Stop followed by native Terminate. A failed
init/start cannot safely use that wrapper path; native termination is attempted
instead. Cleanup errors never hide the original acquisition exception.
"""
from __future__ import annotations


class CameraResource:
    def __init__(self, resource):
        self.resource = resource
        self.init_attempted = False
        self.initialized = False
        self.started = False
        self.closed = False

    def init(self, *args, **kwargs):
        self.init_attempted = True
        result = self.resource.init(*args, **kwargs)
        self.initialized = True
        return result

    def start(self, *args, **kwargs):
        result = self.resource.start(*args, **kwargs)
        self.started = True
        return result

    def close(self):
        if self.closed:
            return []
        self.closed = True
        if not self.init_attempted:
            return []
        errors = []
        if self.started:
            try:
                self.resource.terminate()
                return errors
            except Exception as exc:
                errors.append(str(exc))
        # Use native Terminate directly: wrapper Stop may reject unstarted or
        # already stopped streams and prevent native release from being reached.
        for name in ("_sensor", "_stream", "_native"):
            native = getattr(self.resource, name, None)
            if native is None:
                continue
            operation = getattr(native, "Terminate", None) or getattr(native, "terminate", None)
            if operation is not None:
                try:
                    operation()
                    return errors
                except Exception as exc:
                    errors.append(str(exc))
        if not self.started and not errors:
            errors.append("No native termination entry point for partially initialized SDK resource")
        return errors
