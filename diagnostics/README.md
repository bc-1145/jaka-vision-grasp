# 本地诊断资料

设备发现、历史 IO 映射和安装诊断日志保存在本目录，具体诊断文件保留在本机，不随 Git 仓库上传。

在工作区根目录运行 `gripper_discovery.py --inspect` 可生成 `device_discovery.json`。实际夹爪配置仍是根目录的 `gripper_devices.json`，不会从这里的历史映射自动恢复。

操作说明见 [夹具控制](../docs/README_夹具控制.md)。
