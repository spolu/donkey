UNITY ?= /Applications/Unity/Unity.app/Contents/MacOS/Unity
PYTHON ?= python3

UNITY_FLAGS = -quit -batchmode

# Target OS detection
UNAME := $(shell uname -s)
ifeq ($(UNAME),Linux)
  UNITY_FLAGS += -executeMethod BuildPlayer.PerformBuildLinux64
endif
ifeq ($(UNAME),Darwin)
  UNITY_FLAGS += -executeMethod BuildPlayer.PerformBuildOSX
endif

SIM_PATH = $(abspath sim)

simulation:
	mkdir -p build
	$(UNITY) -projectPath $(SIM_PATH) $(UNITY_FLAGS)

clean:
	rm -rf build/*
