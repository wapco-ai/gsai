import tensorflow
from tensorflow.python.client import device_lib
print("\n📡 Devices:")
for d in device_lib.list_local_devices():
    print(f"{d.name} - {d.device_type} - {round(d.memory_limit / (1024**2))}MB")
