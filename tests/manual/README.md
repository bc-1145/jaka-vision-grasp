# Manual hardware tests

Run from the project root:

```powershell
.\.camera-runtime\python.exe tests/manual/test_camera.py
python tests/manual/test_jaka.py
```

The camera test opens the camera streams. The JAKA test powers on and enables the robot, then commands a linear move to the pose in the script. These scripts run only when executed directly. SDK files remain in their original folders.
