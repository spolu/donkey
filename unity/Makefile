PYTHON ?= python3

UNITY_FLAGS = -quit -batchmode -nographics

# Target OS detection
UNAME := $(shell uname -s)
ifeq ($(UNAME),Linux)
  UNITY ?= ~/opt/unity3d/Editor/Unity
  UNITY_FLAGS += -serial $(UNITY_SERIAL) -executeMethod BuildPlayer.PerformBuildLinux64
endif
ifeq ($(UNAME),Darwin)
  UNITY ?= /Applications/Unity/Unity.app/Contents/MacOS/Unity
  UNITY_FLAGS += -executeMethod BuildPlayer.PerformBuildOSX
endif

PROJECT_PATH = $(abspath sim)

simulation:
	mkdir -p build
	$(UNITY) -projectPath $(PROJECT_PATH) $(UNITY_FLAGS)

clean:
	rm -rf build/*
