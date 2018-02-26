PYTHON ?= python3

UNITY_FLAGS = -quit -batchmode

# Target OS detection
UNAME := $(shell uname -s)
ifeq ($(UNAME),Linux)
  UNITY ?= DISPLAY=:1 ~/opt/unity3d/Editor/Unity
  UNITY_FLAGS += -executeMethod BuildPlayer.PerformBuildLinux64
  SIM_PATH = $(abspath ./build/sim)
endif
ifeq ($(UNAME),Darwin)
  UNITY ?= /Applications/Unity/Unity.app/Contents/MacOS/Unity
  UNITY_FLAGS += -executeMethod BuildPlayer.PerformBuildOSX
  SIM_PATH = $(abspath ./build/sim.app/Contents/MacOS/sim)
endif

PROJECT_PATH = $(abspath sim)

simulation:
	mkdir -p build
	$(UNITY) -projectPath $(PROJECT_PATH) $(UNITY_FLAGS)

clean:
	rm -rf build/*
