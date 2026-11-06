# Build Docker image
.PHONY: docker_build
docker_build:
	docker build -t xadaptctrl .

# Run Docker container with robot-related environment (ROS, etc.)
.PHONY: docker_run_sim
docker_run:
	docker run -it --network host   --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --device=/dev/video0:/dev/video0 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --entrypoint /bin/bash xadaptctrl

# Compiled GeneralCode trirotor sim (Python module `GeneralCode`, used by simulateC.py).
# No git here: third_party/GeneralCode must already be checked out (see README).
PYTHON ?= python
SIM_BUILD := trirotor_sim/build

.PHONY: sim
sim:
	@test -f third_party/GeneralCode/Components/CMakeLists.txt || \
		(echo "third_party/GeneralCode is missing: initialise the submodule first (see README)"; exit 1)
	cmake -S trirotor_sim -B $(SIM_BUILD) -DCMAKE_BUILD_TYPE=Release \
		-DPython3_EXECUTABLE=$$(command -v $(PYTHON)) \
		-Dpybind11_DIR=$$($(PYTHON) -m pybind11 --cmakedir)
	cmake --build $(SIM_BUILD) -j

.PHONY: sim_clean
sim_clean:
	rm -rf $(SIM_BUILD)