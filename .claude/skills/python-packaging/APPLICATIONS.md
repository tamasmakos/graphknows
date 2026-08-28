# Shipping an application

Work backward from the target environment. Each rung down assumes less of the target and ships more of it — which means a bigger artifact and harder security updates, because a bundled component can only be patched by rebuilding.

| The target already has | Ship |
|---|---|
| A framework or PaaS | Whatever its own deploy guide says (Kivy, Beeware, Brython, Flexx for frontend/mobile) |
| A working Python | zipapp, PEX, or shiv — shiv gives the smallest output |
| A separate ecosystem | conda/Anaconda, ActivePython, WinPython — the answer when you need OS-level deps |
| An OS, no Python | Freeze it: PyInstaller, cx_Freeze, py2exe (Win), py2app (Mac), pynsist, constructor, osnap |
| A kernel | A container: Docker, Flatpak, Snapcraft, AppImage |
| Hardware | A VM image: KVM, Hyper-V, VHD/AMI, OpenStack |
| Nothing | CircuitPython/MicroPython on the device |

OS-native packages (deb/RPM) sit alongside these; FPM generates both.

For production, do not deploy by running `pip install` from the internet into a virtualenv — pin and vendor, or build an image.
